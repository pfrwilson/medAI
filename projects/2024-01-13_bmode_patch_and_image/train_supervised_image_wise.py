from medAI.datasets.nct2013.nctbmode1024px import (
    BModeImages,
    BModePatches,
    CoresSelector,
    PatientSelector,
)
import torch
from tqdm import tqdm
import wandb
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from simple_parsing import parse, ArgumentParser, choice, subgroups
from dataclasses import dataclass
import backbone_models
from medAI.utils import set_seed
from torch import nn
from typing import Literal
import argparse
import rich_argparse
from abc import ABC, abstractmethod
from medAI.utils.registry import FactoryRegistry
import sys
import logging 
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# detect anomalies
torch.autograd.set_detect_anomaly(True)
import os


model_registry = FactoryRegistry(name="Model", desc="Model used in the experiment.")


@model_registry.register(name="sliding_window")
def build_sliding_window_model(
    model_name: Literal[*tuple(backbone_models.__all__)] = "resnet10t",
    patch_size: int = 128,
    stride: int = 32,
    needle_mask_threshold: float = 0.5,
    prostate_mask_threshold: float = -1,
):
    model = getattr(backbone_models, model_name)()
    model = SlidingWindowModel(
        model,
        kernel_size=(patch_size, patch_size),
        stride=(stride, stride),
        needle_mask_threshold=needle_mask_threshold,
        prostate_mask_threshold=prostate_mask_threshold,
    )
    return model


@model_registry.register(name="convolutional")
def build_convolutional_model(
    needle_mask_threshold: float = 0.5,
    prostate_mask_threshold: float = -1,
    pos_weight: float = 1.0,
):
    model = ConvolutionalDetectionModel(
        needle_mask_threshold=needle_mask_threshold,
        prostate_mask_threshold=prostate_mask_threshold,
        pos_weight=pos_weight,
    )
    return model


@model_registry.register(name="segmentation")
def build_segmentation_model(
    segmentation_backbone: Literal["unet", "medsam"] = "unet",
    segmentation_backbone_checkpoint: str = None,
    needle_mask_threshold: float = 0.5,
    prostate_mask_threshold: float = -1,
    pos_weight: float = 1.0,
):
    if segmentation_backbone == "unet":
        from monai.networks.nets.unet import UNet

        model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
        )
    elif segmentation_backbone == "medsam":
        from medAI.modeling.sam import MedSAMForFinetuning

        model = MedSAMForFinetuning(freeze_backbone=False)
        torch.compile(model)
    if segmentation_backbone_checkpoint is not None:
        model.load_state_dict(torch.load(segmentation_backbone_checkpoint))
    return SemanticSegmentationDetector(
        model,
        needle_mask_threshold=needle_mask_threshold,
        prostate_mask_threshold=prostate_mask_threshold,
        pos_weight=pos_weight,
    )


def parse_args():
    class HelpFormatter(
        rich_argparse.ArgumentDefaultsRichHelpFormatter,
        rich_argparse.MetavarTypeRichHelpFormatter,
    ):
        ...

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter, add_help=False, description=__doc__
    )
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--name", type=str, default="supervised_image_wise")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--benign-cancer-ratio-train",
        type=float,
        default=1.0,
        help="Ratio of benign to cancer cores in the training set",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="batch size for training"
    )
    parser.add_argument(
        "--min-pct-cancer-train",
        type=int,
        default=40,
        help="Minimum percentage of cancer in the training set",
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        default="none",
        choices=["none", "v1"],
        help="Which augmentations to apply to the training set",
    )


    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "reduce_on_plateau", "warmup_cosine"],
        help="Which learning rate scheduler to use",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0, help="Weight decay")

    model_registry.add_argparse_args(parser)
    PatientSelector.add_arguments(parser)

    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--compile-model", action="store_true", help="Compile model (torch 2.0)", default=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-h", action="help", help="show this help message and exit")
    args = parser.parse_args()

    return args


def main(args):
    wandb.init(project="2024-01-13_bmode_patch_and_image", config=args, name=args.name)

    set_seed(args.seed)

    patient_selector = PatientSelector.from_argparse_args(args)

    (
        train_loader_image,
        train_loader_image_for_eval,
        val_loader_image,
        test_loader_image,
    ) = dataloaders(
        patient_selector,
        benign_cancer_ratio_train=args.benign_cancer_ratio_train,
        augmentations=args.augmentations,
        batch_size=args.batch_size,
    )

    model = model_registry.build_object_from_argparse_args(args)
    model.cuda()
    logging.info(f'Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')
    if args.compile_model:
        logging.info("Compiling model")
        torch.compile(model)

    from torch import optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from medAI.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
    elif args.scheduler == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            5,
            args.epochs,
            warmup_start_lr=1e-9,
            eta_min=1e-7,
        )

    best_score = 0
    for epoch in range(0, args.epochs):
        print(f"Epoch {epoch}")
        wandb.log({"lr": scheduler.get_last_lr()[0]})

        train_loop(
            train_loader_image,
            model,
            optimizer,
        )

        evaluation_loop(
            train_loader_image_for_eval,
            model,
            log_prefix="train_",
            log_image_freq=100,
        )
        val_metrics = evaluation_loop(
            val_loader_image, model, log_prefix="val_", log_image_freq=100
        )
        if val_metrics["auc_high_involvement"] > best_score:
            print("New best score!")
            best_score = val_metrics["auc_high_involvement"]
            evaluation_loop(test_loader_image, model, log_prefix="test_")
        scheduler.step(
            val_metrics["auc_high_involvement"]
        ) if args.scheduler == "reduce_on_plateau" else scheduler.step()


def train_loop(
    loader,
    model,
    optimizer,
    # needle_mask_threshold=0.5,
    # prostate_mask_threshold=-1,
):
    model.train()
    for i, batch in enumerate(tqdm(loader, desc="Training")):
        image = batch["bmode"]
        label = batch["label"]
        needle_mask = batch["needle_mask"]
        prostate_mask = batch["prostate_mask"]

        image = image.cuda()
        needle_mask = needle_mask.cuda()
        prostate_mask = prostate_mask.cuda()
        label = label.cuda()

        # masks = [needle_mask, prostate_mask]
        # thresholds = [needle_mask_threshold, prostate_mask_threshold]
        # patches, label = MaskedSlidingWindow(kernel_size=(128, 128), stride=(32, 32))(
        #     image, masks, thresholds, label
        # )
        # label = label.unsqueeze(-1).float()
        #
        # preds = model(patches)
        #
        # loss = criterion(preds, label)
        loss = model.get_loss(image, needle_mask, prostate_mask, label)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"loss": loss.item()})


@torch.no_grad()
def evaluation_loop(loader, model, log_prefix="", log_image_freq=10):
    model.eval()

    preds = []
    labels = []
    involvement = []

    for i, batch in enumerate(tqdm(loader)):
        image = batch["bmode"]
        label = batch["label"]
        needle_mask = batch["needle_mask"]
        image = image.cuda()
        needle_mask = needle_mask.cuda()

        label = label.cuda()
        heatmap = model(image)
        needle_mask = torch.nn.functional.interpolate(
            needle_mask, size=heatmap.shape[-2:]
        )
        needle_mask_flat = rearrange(needle_mask, "b c h w -> b (h w) c")[..., 0]
        needle_mask_flat = needle_mask_flat > 0.5

        heatmap_flattened = rearrange(heatmap, "b c h w -> b (h w) c")[..., 0]
        needle_heatmap = heatmap_flattened[needle_mask_flat]
        needle_heatmap = needle_heatmap.mean(dim=-1)

        if log_image_freq is not None and i % log_image_freq == 0:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image[0, 0].cpu().numpy(), cmap="gray", extent=[0, 46, 28, 0])
            ax[0].set_title("Image")
            ax[1].imshow(
                heatmap[0, 0].cpu().numpy(), extent=[0, 46, 28, 0], vmin=0, vmax=1
            )
            ax[1].set_title(f"Heatmap (GT {label[0].item()})")
            ax[1].imshow(
                needle_mask[0, 0].cpu().numpy(),
                alpha=0.5 * needle_mask[0, 0].cpu().numpy(),
                extent=[0, 46, 28, 0],
                cmap="Reds",
            )
            tag = batch["tag"]
            grade = batch["grade"]
            wandb.log(
                {
                    f"{log_prefix}heatmap": wandb.Image(
                        fig, caption=f"{tag[0]} GT {grade[0]}"
                    )
                }
            )
            plt.close()

        preds.append(needle_heatmap)
        labels.append(label)
        involvement.append(batch["pct_cancer"])

    preds = torch.stack(preds).cpu()
    labels = torch.cat(labels).cpu()
    involvement = torch.cat(involvement).cpu()

    high_involvement = (involvement > 40) | (labels == 0)

    from sklearn.metrics import roc_auc_score

    metrics = {}
    metrics["auc"] = roc_auc_score(labels, preds)
    metrics["auc_high_involvement"] = roc_auc_score(
        labels[high_involvement], preds[high_involvement]
    )

    wandb.log({f"{log_prefix}{k}": v for k, v in metrics.items()})
    return metrics


def dataloaders(
    patient_selector,
    benign_cancer_ratio_train=1.0,
    augmentations="none",
    batch_size=4,
    min_pct_cancer_train=40,
):
    def transform_image(item, augmentations="none"):
        from torchvision.tv_tensors import Image, Mask
        from torchvision.transforms import v2 as T

        image = item["bmode"]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        image = image.repeat(3, 1, 1)
        image = image / 255.0
        image = Image(image)
        item["bmode"] = image
        C, H, W = image.shape

        needle_mask = item["needle_mask"]
        needle_mask = torch.tensor(needle_mask, dtype=torch.float32).unsqueeze(0)
        needle_mask = Mask(needle_mask)
        needle_mask = T.Resize((H, W))(needle_mask)
        item["needle_mask"] = needle_mask

        prostate_mask = item["prostate_mask"]
        prostate_mask = torch.tensor(prostate_mask, dtype=torch.float32).unsqueeze(0)
        prostate_mask = Mask(prostate_mask)
        prostate_mask = T.Resize((H, W))(prostate_mask)
        item["prostate_mask"] = prostate_mask

        if augmentations == "v1":
            aug = T.RandomApply(
                [T.RandomAffine(degrees=0, translate=(0.2, 0.2))],
                p=1,
            )
            item["bmode"], item["needle_mask"], item["prostate_mask"] = aug(
                item["bmode"], item["needle_mask"], item["prostate_mask"]
            )

        label = torch.tensor(item["grade"] != "Benign", dtype=torch.long)
        item["label"] = label

        return item

    cores_selector_train = CoresSelector(
        patient_selector=patient_selector,
        min_involvement=min_pct_cancer_train,
        remove_benign_from_positive_patients=True,
        benign_to_cancer_ratio=benign_cancer_ratio_train,
    )
    cores_selector_eval = CoresSelector(
        patient_selector=patient_selector,
    )

    train_dataset_image = BModeImages(
        split="train",
        cores_selector=cores_selector_train,
        transform=lambda item: transform_image(item, augmentations=augmentations),
    )
    train_loader_image = torch.utils.data.DataLoader(
        train_dataset_image, batch_size=batch_size, shuffle=True, num_workers=4
    )
    train_loader_image_for_eval = torch.utils.data.DataLoader(
        train_dataset_image, batch_size=1, shuffle=False, num_workers=4
    )

    val_dataset_image = BModeImages(
        split="val",
        cores_selector=cores_selector_eval,
        transform=transform_image,
    )
    val_loader_image = torch.utils.data.DataLoader(
        val_dataset_image, batch_size=1, shuffle=False, num_workers=4
    )

    test_dataset_image = BModeImages(
        split="test",
        cores_selector=cores_selector_eval,
        transform=transform_image,
    )
    test_loader_image = torch.utils.data.DataLoader(
        test_dataset_image, batch_size=1, shuffle=False, num_workers=4
    )

    return (
        train_loader_image,
        train_loader_image_for_eval,
        val_loader_image,
        test_loader_image,
    )


# Sliding Window heatmap approach
class SlidingWindowModel(torch.nn.Module):
    """Wraps a patchwise model such that it can be applied to a whole image, to generate a heatmap."""

    def __init__(
        self,
        window_model,
        kernel_size,
        stride,
        needle_mask_threshold=0.5,
        prostate_mask_threshold=-1,
    ):
        super().__init__()
        self.window_model = window_model
        self.kernel_size = kernel_size
        self.stride = stride
        self.needle_mask_threshold = needle_mask_threshold
        self.prostate_mask_threshold = prostate_mask_threshold

    def forward(self, image):
        H, W = image.shape[-2:]
        kH, kW = self.kernel_size
        sH, sW = self.stride

        nH = n_windows(H, kH, sH)
        nW = n_windows(W, kW, sW)

        unfold = torch.nn.Unfold(kernel_size=(kH, kW), stride=(sH, sW))
        patches = unfold(image)
        patches = rearrange(
            patches,
            "b (c k1 k2) (nh nw) -> (b nh nw) c k1 k2",
            c=1,
            k1=kH,
            k2=kW,
            nh=nH,
            nw=nW,
        )
        predictions = self.window_model(patches)

        B, C = predictions.shape
        if C == 1:
            probs = predictions.sigmoid()
        else:
            probs = predictions.softmax(dim=1)[:, [1]]

        probs = repeat(probs, "b c -> b c k1 k2", c=1, k1=kH, k2=kW)
        probs = rearrange(
            probs,
            "(b nh nw) c k1 k2 -> b (c k1 k2) (nh nw)",
            nh=nH,
            nw=nW,
            k1=kH,
            k2=kW,
        )

        fold = torch.nn.Fold(output_size=(H, W), kernel_size=(kH, kW), stride=(sH, sW))
        n_contributions = fold(torch.ones_like(probs))
        hmap = fold(probs) / n_contributions

        return hmap

    def get_loss(self, image, needle_mask, prostate_mask, label):
        masks = [needle_mask, prostate_mask]
        thresholds = [self.needle_mask_threshold, self.prostate_mask_threshold]
        patches, label, batch_idx = MaskedSlidingWindow(
            kernel_size=(128, 128), stride=(32, 32)
        )(image, masks, thresholds, label)
        label = label.unsqueeze(-1).float()

        preds = self.window_model(patches)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, label)

        return loss


class MaskedSlidingWindow(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        self.unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, image, masks, thresholds, label):
        B, C, H, W = image.shape

        kH, kW = self.kernel_size
        sH, sW = self.stride

        nH = n_windows(H, kH, sH)
        nW = n_windows(W, kW, sW)

        patches = self.unfold(image)

        patches = rearrange(
            patches,
            "b (c k1 k2) (nh nw) -> (b nh nw) c k1 k2",
            c=1,
            k1=kH,
            k2=kW,
            nh=nH,
            nw=nW,
        )

        labels = repeat(label, "b -> (b nh nw)", nh=nH, nw=nW)
        batch_idx = torch.arange(B)
        batch_idx = repeat(batch_idx, "b -> (b nh nw)", nh=nH, nw=nW)

        mask_patches_list = []
        for mask, threshold in zip(masks, thresholds):
            mask = torch.nn.functional.interpolate(mask, size=(H, W))
            mask_patches = self.unfold(mask)

            mask_patches = rearrange(
                mask_patches,
                "b (c k1 k2) (nh nw) -> (b nh nw) c k1 k2",
                c=1,
                k1=kH,
                k2=kW,
                nh=nH,
                nw=nW,
            )
            mask_patches = mask_patches.mean(
                dim=(-1, -2, -3)
            )  # average of mask values in patch
            mask_patches = mask_patches > threshold

            mask_patches_list.append(mask_patches)

        mask_patches = torch.stack(mask_patches_list).all(dim=0)

        patches = patches[mask_patches]  # select only patches where mask is > threshold
        labels = labels[mask_patches]
        batch_idx = batch_idx[mask_patches]

        return patches, labels, batch_idx


class MaskedPixelSelection(torch.nn.Module):
    def forward(self, image, masks, thresholds, label):
        B, C, H, W = image.shape

        label = repeat(label, "b -> (b h w)", h=H, w=W)
        batch_idx = torch.arange(B)
        batch_idx = repeat(batch_idx, "b -> (b h w)", h=H, w=W)

        mask_list = []
        for mask, threshold in zip(masks, thresholds):
            mask = torch.nn.functional.interpolate(mask, size=(H, W))
            mask_patches = mask > threshold
            mask_list.append(mask_patches)
        mask = torch.stack(mask_list).all(dim=0)

        label_flat = rearrange(label, "b h w -> (b h w)")
        batch_idx_flat = rearrange(batch_idx, "b h w -> (b h w)")
        image_flat = rearrange(image, "b c h w -> (b h w) c")
        mask_flat = rearrange(mask, "b c h w -> (b h w) c")[..., 0]

        pixels = image_flat[mask_flat]
        labels = label_flat[mask_flat]
        batch_idx = batch_idx_flat[mask_flat]

        return pixels, labels, batch_idx


# Convolutional feature map classification approach
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=2 if downsample else 1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if self.downsample:
            self.downsample_residual = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=2
            )
        else:
            self.downsample_residual = nn.Identity()

    def forward(self, x):
        residual = self.downsample_residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return x


class ChannelNorm(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.new_mean = torch.nn.Parameter(torch.zeros(num_channels))
        self.new_std = torch.nn.Parameter(torch.ones(num_channels))

    def forward(self, X):
        B, C, H, W = X.shape
        X = torch.nn.functional.normalize(X, dim=1, p=2)
        X = X * self.new_std.view(1, -1, 1, 1) + self.new_mean.view(1, -1, 1, 1)
        return X


class ConvolutionalDetectionModel(torch.nn.Module):
    def __init__(
        self, needle_mask_threshold=0.5, prostate_mask_threshold=-1, pos_weight=1.0
    ):
        super().__init__()
        self.needle_mask_threshold = needle_mask_threshold
        self.prostate_mask_threshold = prostate_mask_threshold
        self.pos_weight = pos_weight

        self.conv_backbone = torch.nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Block(32, 64, downsample=True),
            Block(64, 128, downsample=True),
            Block(128, 256, downsample=True),
            Block(256, 512, downsample=True),
        )

        self.detection_head = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, image):
        feature_map = self.conv_backbone(image)
        heatmap = self.detection_head(feature_map)
        return heatmap

    def get_loss(self, image, needle_mask, prostate_mask, label):
        heatmap = self(image)

        label = label.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        label = label.float().repeat(1, 1, *heatmap.shape[-2:])
        batch_idx = torch.arange(image.shape[0], device=image.device).repeat(
            1, 1, *heatmap.shape[-2:]
        )

        needle_mask = torch.nn.functional.interpolate(
            needle_mask, size=heatmap.shape[-2:]
        )
        prostate_mask = torch.nn.functional.interpolate(
            prostate_mask, size=heatmap.shape[-2:]
        )
        mask = (needle_mask > self.needle_mask_threshold) & (
            prostate_mask > self.prostate_mask_threshold
        )

        heatmap = heatmap[mask]
        label = label[mask]
        batch_idx = batch_idx[mask]

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))

        loss = criterion(heatmap, label.float())

        return loss


# Segmentation Models
class SemanticSegmentationDetector(torch.nn.Module):
    def __init__(
        self,
        segmentation_backbone,
        needle_mask_threshold=0.5,
        prostate_mask_threshold=-1,
        pos_weight=1.0,
    ):
        super().__init__()
        self.segmentation_backbone = segmentation_backbone
        self.needle_mask_threshold = needle_mask_threshold
        self.prostate_mask_threshold = prostate_mask_threshold
        self.pos_weight = pos_weight

    def forward(self, image):
        return self.segmentation_backbone(image)

    def get_loss(self, image, needle_mask, prostate_mask, label):
        heatmap = self(image)

        label = label.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        label = label.float().repeat(1, 1, *heatmap.shape[-2:])
        batch_idx = torch.arange(image.shape[0], device=image.device)[
            :, None, None, None
        ].repeat(1, 1, *heatmap.shape[-2:])

        needle_mask = torch.nn.functional.interpolate(
            needle_mask, size=heatmap.shape[-2:]
        )
        prostate_mask = torch.nn.functional.interpolate(
            prostate_mask, size=heatmap.shape[-2:]
        )
        mask = (needle_mask > self.needle_mask_threshold) & (
            prostate_mask > self.prostate_mask_threshold
        )

        heatmap = heatmap[mask]
        label = label[mask]
        batch_idx = batch_idx[mask]

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))

        loss = criterion(heatmap, label.float())

        return loss


if __name__ == "__main__":
    args = parse_args()

    main(args)
