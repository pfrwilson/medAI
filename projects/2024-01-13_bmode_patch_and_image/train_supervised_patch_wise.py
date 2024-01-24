"""Runs the training of the supervised model on BMode patches. 

A patch-wise model inputs a patch and outputs a prediction. To train the model, 
we extract patches from the BMode images and train the model on these patches. The 
patches are labeled with biopsy result of core that is associated with the ultrasound 
image. The model is trained to minimize the cross entropy loss between the predicted
and the ground truth labels.

For evaluation, we use two different protocols: patch-wise evaluation and image-wise
evaluation. In patch-wise evaluation, we extract patches from the BMode images and
evaluate the model on these patches. In image-wise evaluation, we input the entire
BMode image to the model and evaluate the model on the resulting heatmap. The heatmap
is averaged over the needle mask to obtain a single score per image. This is a more
realistic evaluation protocol than the patch-wise evaluation protocol, which uses a
model that inputs a patch and outputs a prediction.

This script should be used as a baseline for studying bmode-based cancer detectors.
It uses the same basic methodology as the RF-based methods which we have previously 
published. 
"""

from argparse import Action
from medAI.datasets.nct2013.nctbmode1024px import (
    BModeImages,
    BModePatches,
    CoresSelector,
    n_windows,
    PatientSelector, 
)
import torch
from tqdm import tqdm
import wandb
from einops import rearrange
import matplotlib.pyplot as plt
from simple_parsing import parse
import argparse
import rich_argparse
from dataclasses import dataclass
import typing as tp
import logging 
from medAI.utils import EarlyStopping
import random 
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


SUPPORTED_MODELS = [
    "resnet10t",
    "resnet10t_instance_norm",
    "resnet10t_gn_instance_norm",
    "resnet10t_gn",
]


def parse_args():

    class HelpFormatter(
        rich_argparse.ArgumentDefaultsRichHelpFormatter,
        rich_argparse.MetavarTypeRichHelpFormatter,
    ):
        ...

    parser = argparse.ArgumentParser(formatter_class=HelpFormatter, add_help=False, description=__doc__)
    parser.add_argument(
        "--name", type=str, default="supervised_patch_wise", help="name of the run"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="number of epochs to train for"
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--batch-size", type=int, default=64, help="batch size for training"
    )
    data.add_argument("--benign-cancer-ratio-train", type=float, default=1.0)
    data.add_argument("--patch-size", type=int, default=128)
    data.add_argument("--stride", type=int, default=64)
    data.add_argument(
        "--needle-mask-threshold",
        type=float,
        default=0.5,
        help="overlap threshold with the needle mask to select training patches",
    )
    data.add_argument(
        "--prostate-mask-threshold",
        type=float,
        default=-1,
        help="overlap threshold with prostate mask to select training patches",
    )
    data.add_argument(
        "--augmentations",
        type=str,
        default="none",
        choices=["none", "v1"],
        help="augmentations to use for training",
    )
    PatientSelector.add_arguments(parser, data)
    
    data.add_argument(
        "--min-involvement-train",
        type=float,
        default=40,
        help="minimum involvement of cancer in a core to be included in the training set",
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet10t_instance_norm",
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    
    parser.add_argument('-h', action='help', help='show this help message and exit')

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(args: argparse.Namespace):
    logging.info(f"Running experiment {args.name}")
    set_seed(args.seed)

    wandb.init(project="2024-01-13_bmode_patch_and_image", config=args, name=args.name)

    patient_selector = PatientSelector.from_argparse_args(args)

    train_loader_patch, val_loader_patch, test_loader_patch = patch_dataloaders(
        patient_selector=patient_selector,
        batch_size=args.batch_size,
        benign_cancer_ratio_train=args.benign_cancer_ratio_train,
        patch_size=args.patch_size,
        stride=args.stride,
        needle_mask_threshold=args.needle_mask_threshold,
        prostate_mask_threshold=args.prostate_mask_threshold,
        augmentations=args.augmentations,
        min_involvement_train=args.min_involvement_train,
    )

    train_loader_image, val_loader_image, test_loader_image = image_dataloaders(patient_selector)

    model, criterion = build_model(args.model_name)
    model.cuda()
    image_model = SlidingWindowModel(
        model,
        kernel_size=(args.patch_size, args.patch_size),
        stride=(args.stride, args.stride),
    )

    from torch import optim
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, args.epochs)

    best_score = 0
    best_model_state = None
    monitor = EarlyStopping(patience=5, mode='max')

    for epoch in range(0, args.epochs):
        print(f"Epoch {epoch}")

        patchwise_train_loop(train_loader_patch, model, criterion, optimizer)
        scheduler.step()
        wandb.log({"lr": scheduler.get_last_lr()[0]})

        patchwise_evaluation_loop(train_loader_patch, model, log_prefix="train_")
        val_metrics = patchwise_evaluation_loop(
            val_loader_patch, model, log_prefix="val_"
        )
        if val_metrics["auc_high_involvement"] > best_score:
            patchwise_evaluation_loop(test_loader_patch, model, log_prefix="test_")
            print("New best score!")
            best_score = val_metrics["auc_high_involvement"]
            best_model_state = model.state_dict()

        monitor(val_metrics["auc_high_involvement"])
        if monitor.early_stop:
            break

    logging.info("Training finished. Now testing with image-wise evaluation.")
    model.load_state_dict(best_model_state)
    evaluation_loop(test_loader_image, image_model, log_prefix="best_test_")


def patchwise_train_loop(loader, model, criterion, optimizer) -> None:
    """Runs a single epoch of patch-wise training.

    Args:
        loader: The dataloader to use. Batches will be a dictionary with keys "patch", "label", "core_id", "involvement"
        model: The model to train. Model will input a patch and output a prediction expected by the criterion.
        criterion: The loss function to use, will be applied to the model's output and the label.
        optimizer: The optimizer to use.
    """

    model.train()
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        patch, label = batch["patch"], batch["label"]
        patch = patch.cuda()
        label = label.cuda()
        logits = model(patch)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        correct += (logits.argmax(dim=-1) == label).sum().item()
        total += len(label)

        wandb.log({"loss": loss.item()})


@torch.no_grad()
def patchwise_evaluation_loop(loader, model, log_prefix=""):
    """Runs a single epoch of patch-wise evaluation.

    Runs the model to create patch wise predictions - aggregates them into corewise
    predictions and computes metrics.

    Args:
        loader: The dataloader to use. Batches will be a dictionary with keys "patch", "label", "core_id", "involvement"
        model: The model to evaluate. Model will input a patch and output a prediction.
        log_prefix: A prefix to use for logging metrics to wandb.
        log_image_freq: How often to log images to wandb. If None, no images will be logged.

    Returns:
        metrics: A dictionary with keys "auc" and "auc_high_involvement".
    """

    model.eval()
    preds = []
    labels = []
    involvements = []
    core_ids = []

    for i, batch in enumerate(tqdm(loader, desc="Patch-wise evaluation")):
        patch, label, core_id, involvement = (
            batch["patch"],
            batch["label"],
            batch["core_id"],
            batch["involvement"],
        )
        patch = patch.cuda()
        label = label.cuda()
        logits = model(patch)
        preds.append(logits[:, 1])
        labels.append(label)
        involvements.append(batch["involvement"])
        core_ids.append(batch["core_id"])

    preds = torch.cat(preds).cpu()
    labels = torch.cat(labels).cpu()
    involvements = torch.cat(involvements).cpu()
    core_ids = torch.cat(core_ids).cpu()

    preds_corewise = []
    labels_corewise = []
    involvement_corewise = []
    for core_id in core_ids.unique():
        preds_corewise.append(preds[core_ids == core_id].mean())
        labels_corewise.append(labels[core_ids == core_id][0])
        involvement_corewise.append(involvements[core_ids == core_id][0])

    preds_corewise = torch.stack(preds_corewise)
    labels_corewise = torch.stack(labels_corewise)
    involvement_corewise = torch.stack(involvement_corewise)

    from sklearn.metrics import roc_auc_score

    roc = roc_auc_score(labels_corewise, preds_corewise)

    high_involvement = (involvement_corewise > 40) | (labels_corewise == 0)
    roc_high_involvement = roc_auc_score(
        labels_corewise[high_involvement], preds_corewise[high_involvement]
    )

    metrics = {}
    metrics["auc"] = roc
    metrics["auc_high_involvement"] = roc_high_involvement

    wandb.log({f"{log_prefix}{k}": v for k, v in metrics.items()})
    return metrics


@torch.no_grad()
def evaluation_loop(loader, model, log_prefix="", log_image_freq=10):
    """Runs image-wise evaluation.

    The image-wise evaluation protocol uses a model that inputs an image and outputs a heatmap.
    The heatmap is then averaged over the needle mask to obtain a single score per image.
    This is a more realistic evaluation protocol than the patch-wise evaluation protocol,
    which uses a model that inputs a patch and outputs a prediction.

    Args:
        loader: The dataloader to use. Batches will a dictionary with keys "bmode", "label", "needle_mask", "prostate_mask", "tag", "grade", "pct_cancer"
        model: The model to evaluate. Model should an image and output a heatmap with a single channel representing probability of cancer.
        log_prefix: A prefix to use for logging metrics to wandb.
        log_image_freq: How often to log images to wandb. If None, no images will be logged.

    Returns:
        metrics: A dictionary with keys "auc" and "auc_high_involvement".

    """

    model.eval()

    preds = []
    labels = []
    involvement = []

    for i, batch in enumerate(tqdm(loader, desc="Evaluation")):
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

        if log_image_freq is not None or i % log_image_freq == 0:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image[0, 0].cpu().numpy(), cmap="gray", extent=[0, 46, 28, 0])
            ax[0].set_title("Image")
            ax[1].imshow(
                heatmap[0, 0].cpu().numpy(), extent=[0, 46, 28, 0], vmin=0, vmax=1
            )
            ax[1].set_title(f"Heatmap (GT {label.item()})")
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
    # histogram 
    for label in labels.unique():
        plt.hist(preds[labels == label], bins=20, alpha=0.5, label=f"GT {label}")
    plt.legend()
    wandb.log({f"{log_prefix}histogram": wandb.Image(plt)})

    high_involvement = (involvement > 40) | (labels == 0)

    from sklearn.metrics import roc_auc_score

    metrics = {}
    metrics["auc"] = roc_auc_score(labels, preds)
    metrics["auc_high_involvement"] = roc_auc_score(
        labels[high_involvement], preds[high_involvement]
    )

    # high involvement histogram
    for label in labels.unique():
        plt.hist(
            preds[(labels == label) & high_involvement],
            bins=20,
            alpha=0.5,
            label=f"GT {label}",
        )
    plt.legend()
    wandb.log({f"{log_prefix}histogram_high_involvement": wandb.Image(plt)})

    wandb.log({f"{log_prefix}{k}": v for k, v in metrics.items()})
    return metrics


def build_model(model_name: SUPPORTED_MODELS):
    match model_name:
        case "resnet10t":
            from timm.models.resnet import resnet10t

            model = resnet10t(in_chans=1, num_classes=2)
            criterion = torch.nn.CrossEntropyLoss()
            return model, criterion

        case "resnet10t_instance_norm":
            from timm.models.resnet import resnet10t

            model = torch.nn.Sequential(
                torch.nn.InstanceNorm2d(1), resnet10t(in_chans=1, num_classes=2)
            )
            criterion = torch.nn.CrossEntropyLoss()
            return model, criterion

        case "resnet10t_gn_instance_norm":
            from timm.models.resnet import resnet10t

            model = resnet10t(
                in_chans=1,
                num_classes=2,
                norm_layer=lambda chans: torch.nn.GroupNorm(
                    num_groups=8, num_channels=chans
                ),
            )
            model = torch.nn.Sequential(torch.nn.InstanceNorm2d(1), model)
            return model, torch.nn.CrossEntropyLoss()

        case "resnet10t_gn":
            from timm.models.resnet import resnet10t

            model = resnet10t(
                in_chans=1,
                num_classes=2,
                norm_layer=lambda chans: torch.nn.GroupNorm(
                    num_groups=8, num_channels=chans
                ),
            )
            return model, torch.nn.CrossEntropyLoss()

        case _:
            raise ValueError(f"Unknown model name {model_name}")


def patch_dataloaders(
    patient_selector,
    batch_size=64,
    benign_cancer_ratio_train=1.0,
    patch_size=128,
    stride=32,
    needle_mask_threshold=0.6,
    prostate_mask_threshold=-1,
    min_involvement_train=None,
    augmentations="none",
):
    class PatchTransform:
        def __init__(self, augmentations="none"):
            self.augmentations = augmentations

        def __call__(self, item):
            patch = item["patch"]
            patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
            patch = patch / 255.0
            label = torch.tensor(item["grade"] != "Benign", dtype=torch.long)

            if self.augmentations == "v1":
                from torchvision import transforms as T

                patch = T.RandomHorizontalFlip(p=0.5)(patch)
                patch = T.RandomApply(
                    [T.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5
                )(patch)

            out = {}
            out["patch"] = patch
            out["core_id"] = item["id"]
            out["involvement"] = item["pct_cancer"]
            out["label"] = label
            return out



    train_dataset = BModePatches(
        split="train",
        cores_selector=CoresSelector(
            patient_selector=patient_selector,
            min_involvement=min_involvement_train,
            remove_benign_from_positive_patients=True,
            benign_to_cancer_ratio=benign_cancer_ratio_train,
        ), 
        transform=PatchTransform(augmentations),
        needle_mask_threshold=needle_mask_threshold,
        patch_size=patch_size,
        stride=stride,
        prostate_mask_threshold=prostate_mask_threshold,
    )
    val_dataset = BModePatches(
        split="val",
        cores_selector=CoresSelector(
            patient_selector=patient_selector,
            min_involvement=None,
            remove_benign_from_positive_patients=False,
            benign_to_cancer_ratio=None,
        ),
        transform=PatchTransform(),
        needle_mask_threshold=needle_mask_threshold,
        patch_size=patch_size,
        stride=stride,
        prostate_mask_threshold=prostate_mask_threshold,
    )
    test_dataset = BModePatches(
        split="test",
        cores_selector = CoresSelector(
            patient_selector=patient_selector,
            min_involvement=None,
            remove_benign_from_positive_patients=False,
            benign_to_cancer_ratio=None,
        ), 
        transform=PatchTransform(),
        needle_mask_threshold=needle_mask_threshold,
        patch_size=patch_size,
        stride=stride,
        prostate_mask_threshold=prostate_mask_threshold,
    )

    fig, ax = plt.subplots(3, 3)
    for i in range(9):
        train_dataset.show_patch_extraction(ax[i // 3, i % 3])

    if wandb.run is not None:
        wandb.log({"patch_extraction": wandb.Image(fig)})

    train_loader_patch = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader_patch = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader_patch = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader_patch, val_loader_patch, test_loader_patch


def image_dataloaders(
    patient_selector,
):
    def transform_image(item):
        image = item["bmode"]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        image = image / 255.0
        item["bmode"] = image

        needle_mask = item["needle_mask"]
        needle_mask = torch.tensor(needle_mask, dtype=torch.float32).unsqueeze(0)
        item["needle_mask"] = needle_mask

        prostate_mask = item["prostate_mask"]
        prostate_mask = torch.tensor(prostate_mask, dtype=torch.float32).unsqueeze(0)
        item["prostate_mask"] = prostate_mask

        label = torch.tensor(item["grade"] != "Benign", dtype=torch.long)
        item["label"] = label

        return item

    cores_selector=CoresSelector(patient_selector=patient_selector)
    train_dataset_image = BModeImages(
        split="train",
        transform=transform_image,
        cores_selector=cores_selector,
    )
    train_loader_image = torch.utils.data.DataLoader(
        train_dataset_image, batch_size=1, shuffle=True, num_workers=4
    )

    val_dataset_image = BModeImages(
        split="val",
        transform=transform_image,
        cores_selector=cores_selector,
    )
    val_loader_image = torch.utils.data.DataLoader(
        val_dataset_image, batch_size=1, shuffle=False, num_workers=4
    )

    test_dataset_image = BModeImages(
        split="test",
        transform=transform_image,
        cores_selector=cores_selector,
    )
    test_loader_image = torch.utils.data.DataLoader(
        test_dataset_image, batch_size=1, shuffle=False, num_workers=4
    )

    return train_loader_image, val_loader_image, test_loader_image


class SlidingWindowModel(torch.nn.Module):
    def __init__(self, window_model, kernel_size, stride):
        super().__init__()
        self.window_model = window_model
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    def forward(self, image):
        return create_heatmap(self.window_model, image, self.kernel_size, self.stride)


def create_heatmap(
    model,
    image,
    kernel_size,
    stride,
):
    from einops import rearrange, repeat

    H, W = image.shape[-2:]
    kH, kW = kernel_size
    sH, sW = stride

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
    predictions = model(patches)
    probs = predictions.softmax(dim=-1)[:, [1]]
    probs = repeat(probs, "b c -> b c k1 k2", c=1, k1=kH, k2=kW)
    probs = rearrange(
        probs, "(b nh nw) c k1 k2 -> b (c k1 k2) (nh nw)", nh=nH, nw=nW, k1=kH, k2=kW
    )

    fold = torch.nn.Fold(output_size=(H, W), kernel_size=(kH, kW), stride=(sH, sW))
    n_contributions = fold(torch.ones_like(probs))
    hmap = fold(probs) / n_contributions

    return hmap


if __name__ == "__main__":
    args = parse_args()
    main(args)
