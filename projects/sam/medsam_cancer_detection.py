import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass, asdict
from simple_parsing import ArgumentParser, Serializable, choice
from simple_parsing.helpers import Serializable
from medAI.modeling import LayerNorm2d, Patchify
import medAI
from medAI.utils.setup import BasicExperiment, BasicExperimentConfig
from segment_anything import sam_model_registry
from einops import rearrange, repeat
import wandb
from tqdm import tqdm
import submitit
import matplotlib.pyplot as plt
import os
import typing as tp
from abc import ABC, abstractmethod


@dataclass
class Config(BasicExperimentConfig, Serializable):
    """Training configuration"""

    project: str = "medsam_cancer_detection"
    fold: int = 0
    epochs: int = 10
    optimizer: str = choice(
        "adam", "sgdw", default="adam"
    )
    use_augmentation: bool = False
    lr: float = 0.00001
    wd: float = 0.0
    batch_size: int = 1
    model_name: str = choice(
        "medsam_cancer_detector_v2", "medsam_linear", "medsam_cancer_detector_with_adapters", 
        "resnet_sliding_window",
        default="medsam_cancer_detector_v2"
    )
    debug: bool = False
    accumulate_grad_steps: int = 8


class Experiment(BasicExperiment):
    config_class = Config
    config: Config

    @dataclass
    class Output:
        loss: torch.Tensor | None = None
        patch_predictions: torch.Tensor | None = None
        patch_labels: torch.Tensor | None = None
        core_predictions: torch.Tensor | None = None
        core_labels: torch.Tensor | None = None

    def setup(self):
        # logging setup
        super().setup()
        self.setup_data()

        if "experiment.ckpt" in os.listdir(self.ckpt_dir):
            state = torch.load(os.path.join(self.ckpt_dir, "experiment.ckpt"))
        else:
            state = None

        # Setup model
        match self.config.model_name:
            case "medsam_cancer_detector_v2":
                self.model = MedSAMCancerDetectorV2()
            case "medsam_linear":
                self.model = LinearLayerMedsamCancerDetector()
            case "medsam_cancer_detector_with_adapters":
                self.model = MedSAMCancerDetectorWithAdapters(adapter_dim=128, thaw_patch_embed=True)
            case "resnet_sliding_window":
                self.model = ResnetSlidingWindowCancerDetector()
            case _:
                raise ValueError(f"Unknown model name: {self.config.model_name}")
        self.model = self.model.cuda()
        if state is not None:
            self.model.load_state_dict(state["model"])

        match self.config.optimizer:
            case "adam":
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.config.lr,
                    weight_decay=self.config.wd,
                )
            case "sgdw":
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.config.lr,
                    momentum=0.9,
                    weight_decay=self.config.wd,
                )
        self.scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=5 * len(self.train_loader),
            max_epochs=self.config.epochs * len(self.train_loader),
        )

        if state is not None:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])

        print("Number of parameters: ", sum(p.numel() for p in self.model.parameters()))
        print(
            "Trainable parameters: ",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )
        self.epoch = 0 if state is None else state["epoch"]
        self.best_score = 0 if state is None else state["best_score"]

    def save(self):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
            },
            os.path.join(self.ckpt_dir, "experiment.ckpt"),
        )

    def setup_data(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import v2 as T
        from torchvision.datapoints import Image, Mask

        class Transform:
            def __init__(self, augment=False):
                self.augment = augment
            
            def __call__(self, item):
                bmode = item["bmode"]
                bmode = T.ToTensor()(bmode)
                bmode = T.Resize((1024, 1024), antialias=True)(bmode)
                bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
                bmode = bmode.repeat(3, 1, 1)
                bmode = Image(bmode)

                needle_mask = item["needle_mask"]
                needle_mask = T.ToTensor()(needle_mask).float() * 255
                needle_mask = T.Resize(
                    (1024, 1024), antialias=False, interpolation=InterpolationMode.NEAREST
                )(needle_mask)
                needle_mask = Mask(needle_mask)

                prostate_mask = item["prostate_mask"]
                prostate_mask = T.ToTensor()(prostate_mask).float() * 255
                prostate_mask = T.Resize(
                    (1024, 1024), antialias=False, interpolation=InterpolationMode.NEAREST
                )(prostate_mask)
                prostate_mask = Mask(prostate_mask)

                if self.augment:
                    bmode, needle_mask, prostate_mask = T.RandomAffine(
                        degrees=0, translate=(0.1, 0.1)
                    )(bmode, needle_mask, prostate_mask)

                label = torch.tensor(item["grade"] != "Benign").long()
                return bmode, needle_mask, prostate_mask, label

        from medAI.datasets import ExactNCT2013BModeImages, CohortSelectionOptions, ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation

        train_ds = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split="train",
            transform=Transform(augment=self.config.use_augmentation),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=1,
                min_involvement=40,
                remove_benign_from_positive_patients=True,
                fold=self.config.fold,
            ),
        )
        test_ds = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split="test",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=None, min_involvement=40, fold=self.config.fold
            ),
        )
        if self.config.debug:
            train_ds = torch.utils.data.Subset(train_ds, torch.arange(0, 100))
            test_ds = torch.utils.data.Subset(test_ds, torch.arange(0, 100))

        self.train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )

    def __call__(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.epochs):
            print(f"Epoch {self.epoch}")
            self.run_epoch(self.train_loader, train=True, desc="train")
            self.run_epoch(self.test_loader, train=False, desc="test")

    def run_epoch(self, loader, train=True, desc="train"):
        with torch.no_grad() if not train else torch.enable_grad():
            self.model.train() if train else self.model.eval()

            core_probs = []
            core_labels = []
            patch_probs = []
            patch_labels = []

            acc_steps = 1
            for i, batch in enumerate(tqdm(loader, desc=desc)):
                bmode, needle_mask, prostate_mask, label = batch
                bmode = bmode.cuda()
                needle_mask = needle_mask.cuda()
                prostate_mask = prostate_mask.cuda()
                label = label.cuda()
                out: self.Output = self.model(bmode, needle_mask, label, prostate_mask=prostate_mask)
                loss = out.loss
                
                if train:
                    loss.backward()
                    if acc_steps % self.config.accumulate_grad_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        acc_steps = 1
                    else:
                        acc_steps += 1
                    self.scheduler.step()
                    wandb.log({"train_loss": loss, "lr": self.scheduler.get_last_lr()[0]})

                core_probs.append(out.core_predictions.sigmoid().detach().cpu())
                core_labels.append(out.core_labels.detach().cpu())
                patch_probs.append(out.patch_predictions.sigmoid().detach().cpu())
                patch_labels.append(out.patch_labels.detach().cpu())

                interval = 100 if train else 10
                if i % interval == 0:
                    self.model.show_example(bmode, needle_mask, label, prostate_mask=prostate_mask)
                    wandb.log({f"{desc}_example": wandb.Image(plt)})
                    plt.close()

            core_probs = torch.cat(core_probs)
            core_labels = torch.cat(core_labels)
            plt.hist(core_probs[core_labels == 0], bins=100, alpha=0.5, density=True)
            plt.hist(core_probs[core_labels == 1], bins=100, alpha=0.5, density=True)
            wandb.log({f"{desc}_histogram": wandb.Image(plt)})
            plt.close()

            patch_probs = torch.cat(patch_probs)
            patch_labels = torch.cat(patch_labels)

            from sklearn.metrics import roc_auc_score, balanced_accuracy_score

            core_auc = roc_auc_score(core_labels, core_probs)
            patch_auc = roc_auc_score(patch_labels, patch_probs)
            # core_acc = balanced_accuracy_score(core_labels, core_probs.argmax(dim=-1))
            # patch_acc = balanced_accuracy_score(patch_labels, patch_probs.argmax(dim=-1))

            wandb.log(
                {
                    f"{desc}_core_auc": core_auc,
                    f"{desc}_patch_auc": patch_auc,
                    "epoch": self.epoch,
                    # "train_core_acc": core_acc,
                    # "train_patch_acc": patch_acc,
                }
            )

    def checkpoint(self):
        self.save()
        return super().checkpoint()


@dataclass
class CancerDetectorOutput: 
    loss: torch.Tensor | None = None
    core_predictions: torch.Tensor | None = None
    patch_predictions: torch.Tensor | None = None
    patch_labels: torch.Tensor | None = None
    core_labels: torch.Tensor | None = None
    cancer_logits_map: torch.Tensor | None = None


class CancerDetectorBase(nn.Module, ABC): 
    @abstractmethod
    def forward(self, image, needle_mask, label, prostate_mask) -> CancerDetectorOutput:        
        ...

    @torch.no_grad()
    def show_example(self, image, needle_mask, label, prostate_mask=None):
        import matplotlib.pyplot as plt
        import numpy as np

        output: CancerDetectorOutput = self(image, needle_mask, label, prostate_mask=prostate_mask)

        logits = output.cancer_logits_map
        pred = logits.sigmoid()
        needle_mask = needle_mask.cpu()
        logits = logits.cpu()
        pred = pred.cpu()
        image = image.cpu()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        [ax.set_axis_off() for ax in ax.flatten()]
        kwargs = dict(vmin=0, vmax=1, extent=(0, 46, 0, 28))
        ax[0].imshow(image[0].permute(1, 2, 0), **kwargs)
        if prostate_mask is not None: 
            prostate_mask = prostate_mask.cpu()
            ax[0].imshow(prostate_mask[0, 0], alpha=prostate_mask[0][0] * 0.3, cmap="Blues", **kwargs)
        ax[0].imshow(needle_mask[0, 0], alpha=needle_mask[0][0], cmap="Reds", **kwargs)
        ax[1].imshow(pred[0, 0], **kwargs)
        ax[1].set_title(f"label: {label[0].item()}")


class MedSAMCancerDetectorV2(CancerDetectorBase):
    def __init__(self):
        super().__init__()
        self.medsam_model = sam_model_registry["vit_b"](
            checkpoint="/scratch/ssd004/scratch/pwilson/medsam_vit_b_cpu.pth"
        )

    def get_logits(self, image):
        image_emb = self.medsam_model.image_encoder(image)
        sparse_emb, dense_emb = self.medsam_model.prompt_encoder(None, None, None)
        mask_logits = self.medsam_model.mask_decoder.forward(
            image_emb,
            self.medsam_model.prompt_encoder.get_dense_pe(),
            sparse_emb,
            dense_emb,
            False,
        )[0]
        return mask_logits

    def forward(self, image, needle_mask, label, prostate_mask=None):
        # computes the loss for the model.
        logits_map = self.get_logits(image)
        B, C, H, W = logits_map.shape
        needle_mask = rearrange(
            needle_mask, "b c (nh h) (nw w) -> b c nh nw h w", nh=H, nw=W
        )
        needle_mask = needle_mask.mean(dim=(-1, -2)) > 0.5
        mask = needle_mask

        if prostate_mask is not None: 
            prostate_mask = rearrange(
                prostate_mask, "b c (nh h) (nw w) -> b c nh nw h w", nh=H, nw=W
            )
            prostate_mask = prostate_mask.mean(dim=(-1, -2)) > 0.5
            mask = mask & prostate_mask
            if mask.sum() == 0:
                mask = needle_mask

        core_idx = torch.arange(B, device=logits_map.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)
        label_rep = repeat(label, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[
            ..., 0
        ].bool()
        label_flattened = rearrange(label_rep, "b h w -> (b h w)", h=H, w=W)[
            ..., None
        ].float()
        logits_flattened = rearrange(logits_map, "b c h w -> (b h w) c", h=H, w=W)

        logits = logits_flattened[mask_flattened]
        label = label_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_predictions = logits.sigmoid()
        patch_labels = label
        core_predictions = []
        core_labels = []
        for i in range(B):
            core_idx_i = core_idx == i
            logits_i = logits[core_idx_i]
            predictions_i = logits_i.sigmoid().mean(dim=0)
            core_predictions.append(predictions_i)
            core_labels.append(label[core_idx_i][0])
        core_predictions = torch.stack(core_predictions)
        core_labels = torch.stack(core_labels)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, label.float()
        )

        return CancerDetectorOutput(
            loss=loss,
            core_predictions=core_predictions,
            core_labels=core_labels,
            cancer_logits_map=logits_map, 
            patch_predictions=patch_predictions,
            patch_labels=patch_labels
        )


class LinearLayerMedsamCancerDetector(MedSAMCancerDetectorV2):
    def __init__(self):
        super().__init__()
        self.clf = torch.nn.Conv2d(256, 1, kernel_size=1, padding=0)

    def get_logits(self, image):
        image_emb = self.medsam_model.image_encoder(image)
        logits = self.clf(image_emb)
        return logits


class MedSAMCancerDetectorWithAdapters(MedSAMCancerDetectorV2): 
    def __init__(self, adapter_dim: int, thaw_patch_embed: bool = False):
        super().__init__()
        from models import wrap_image_encoder_with_adapter, freeze_non_adapter_layers
        wrap_image_encoder_with_adapter(self.medsam_model.image_encoder, adapter_dim=adapter_dim)
        freeze_non_adapter_layers(self.medsam_model.image_encoder)
        if thaw_patch_embed:
            for param in self.medsam_model.image_encoder.patch_embed.parameters(): 
                param.requires_grad = True


RESNET10_PATH = '/ssd005/projects/exactvu_pca/checkpoint_store/vicreg_resnet10_feature_extractor_fold0.pth'

class LinearEval(torch.nn.Module):
    def __init__(self, model, linear_layer, freeze_model=True): 
        super().__init__()
        self.model = model
        self.linear_layer = linear_layer
        self.freeze_model = freeze_model

    def forward(self, x):
        with torch.no_grad() if self.freeze_model else torch.enable_grad(): 
            x = self.model(x)
        x = self.linear_layer(x)
        return x

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_model:
            self.model.eval()


class ResnetSlidingWindowCancerDetector(CancerDetectorBase): 
    def __init__(self, freeze_model=True): 
        super().__init__()
        from trusnet.modeling.registry import resnet10_feature_extractor
        feature_extractor = resnet10_feature_extractor().cuda()
        feature_extractor.load_state_dict(torch.load(RESNET10_PATH))
        linear_layer = torch.nn.Linear(512, 1).cuda()
        self.model = LinearEval(feature_extractor, linear_layer, freeze_model=freeze_model)


    def forward(self, image, needle_mask, label, prostate_mask) -> CancerDetectorOutput:
        from medAI.utils import view_as_windows_torch
        
        bmode = image[:, [0], ...] # take only the first channel

        with torch.no_grad(): 
            B, C, H, W = bmode.shape 
            step_size = (int(H / 28), int(W / 46))
            window_size = step_size[0] * 5, step_size[1] * 5

            needle_mask = torch.nn.functional.interpolate(needle_mask, size=(H, W), mode='nearest')
            prostate_mask = torch.nn.functional.interpolate(prostate_mask, size=(H, W), mode='nearest')
            
            needle_mask = view_as_windows_torch(needle_mask, window_size, step_size)
            needle_mask = (needle_mask.mean(dim=(4, 5)) > 0.66)
            needle_mask = rearrange(needle_mask, 'b c nh nw -> (b nh nw) c')[..., 0]

            prostate_mask = view_as_windows_torch(prostate_mask, window_size, step_size)
            prostate_mask = (prostate_mask.mean(dim=(4, 5)) > 0.9)
            prostate_mask = rearrange(prostate_mask, 'b c nh nw -> (b nh nw) c')[..., 0]

            mask = needle_mask & prostate_mask
            if mask.sum() == 0:
                mask = needle_mask

            bmode = view_as_windows_torch(bmode, window_size, step_size)
            B, C, nH, nW, H, W = bmode.shape
            bmode = rearrange(bmode, 'b c nh nw h w -> (b nh nw) c h w')
            bmode = (bmode - bmode.mean(dim=(-1, -2, -3), keepdim=True)) / bmode.std(dim=(-1, -2, -3), keepdim=True)
            bmode = torch.nn.functional.interpolate(bmode, size=(256, 256), mode='bilinear', align_corners=False)

            label = repeat(label, 'b -> b nh nw', nh=nH, nw=nW)
            label = rearrange(label, 'b nh nw -> (b nh nw)')[mask]
            batch_idx = torch.arange(B, device=bmode.device)
            batch_idx = repeat(batch_idx, 'b -> b nh nw', nh=nH, nw=nW)
            batch_idx = rearrange(batch_idx, 'b nh nw -> (b nh nw)')[mask]

        logits = self.model(bmode)

        logits_map = rearrange(logits, '(b nh nw) c -> b c nh nw', b=B, nh=nH, nw=nW)
        logits = logits[mask]
        patch_predictions = logits.sigmoid()
        label = label[..., None]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label.float())

        # breakpoint()

        core_pred = []
        core_label = []
        for i in range(B): 
            core_pred.append(logits[batch_idx == i].sigmoid().mean(dim=0))
            core_label.append(label[batch_idx == i][0])
        core_pred = torch.stack(core_pred)
        core_label = torch.stack(core_label)

        return CancerDetectorOutput(
            loss=loss,
            core_predictions=core_pred,
            core_labels=core_label,
            cancer_logits_map=logits_map, 
            patch_predictions=patch_predictions,
            patch_labels=label
        )   



if __name__ == "__main__":
    Experiment.submit()
