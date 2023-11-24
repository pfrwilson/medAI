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
import logging
import numpy as np
import typing as tp
from abc import ABC, abstractmethod
from medsam_cancer_detection_v2_model_registry import (
    model_registry,
    MaskedPredictionModule,
)
from typing import Any
import typing as tp


@dataclass
class Config(BasicExperimentConfig):
    """Training configuration"""

    project: str = "medsam_cancer_detection_v3"
    fold: int = 0
    n_folds: int = 5
    benign_cancer_ratio_for_training: float | None = None
    epochs: int = 30
    optimizer: tp.Literal["adam", "sgdw"] = "adam"
    use_augmentation: bool = False
    loss: tp.Literal["basic_ce", "involvement_tolerant_loss"] = "basic_ce"
    lr: float = 0.00001
    wd: float = 0.0
    batch_size: int = 1
    model_config: Any = model_registry.get_simple_parsing_subgroups()
    debug: bool = False
    accumulate_grad_steps: int = 8
    min_involvement_pct_training: float = 40.0
    prostate_threshold: float = 0.5
    needle_threshold: float = 0.5


class Experiment(BasicExperiment):
    config_class = Config
    config: Config

    def setup(self):
        # logging setup
        super().setup()
        self.setup_data()

        if "experiment.ckpt" in os.listdir(self.ckpt_dir):
            state = torch.load(os.path.join(self.ckpt_dir, "experiment.ckpt"))
        else:
            state = None

        # Setup model
        self.model = model_registry.build_from_config(self.config.model_config)
        self.model = self.model.cuda()
        torch.compile(self.model)

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

        self.masked_prediction_module_train = MaskedPredictionModule(
            needle_mask_threshold=self.config.needle_threshold,
            prostate_mask_threshold=self.config.prostate_threshold,
        )
        self.masked_prediction_module_test = MaskedPredictionModule()
        self.gradient_scaler = torch.cuda.amp.GradScaler()

        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        logging.info(
            f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )
        self.epoch = 0 if state is None else state["epoch"]
        self.best_score = 0 if state is None else state["best_score"]

    def setup_data(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import v2 as T
        from torchvision.tv_tensors import Image, Mask

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
                    (1024, 1024),
                    antialias=False,
                    interpolation=InterpolationMode.NEAREST,
                )(needle_mask)
                needle_mask = Mask(needle_mask)

                prostate_mask = item["prostate_mask"]
                prostate_mask = T.ToTensor()(prostate_mask).float() * 255
                prostate_mask = T.Resize(
                    (1024, 1024),
                    antialias=False,
                    interpolation=InterpolationMode.NEAREST,
                )(prostate_mask)
                prostate_mask = Mask(prostate_mask)

                if self.augment:
                    bmode, needle_mask, prostate_mask = T.RandomAffine(
                        degrees=0, translate=(0.2, 0.2)
                    )(bmode, needle_mask, prostate_mask)

                label = torch.tensor(item["grade"] != "Benign").long()
                pct_cancer = item["pct_cancer"]
                if np.isnan(pct_cancer):
                    pct_cancer = 0
                involvement = torch.tensor(pct_cancer / 100).float()
                return bmode, needle_mask, prostate_mask, label, involvement

        from medAI.datasets import (
            ExactNCT2013BModeImages,
            CohortSelectionOptions,
            ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation,
        )

        train_ds = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split="train",
            transform=Transform(augment=self.config.use_augmentation),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=self.config.benign_cancer_ratio_for_training,
                min_involvement=self.config.min_involvement_pct_training,
                remove_benign_from_positive_patients=True,
                fold=self.config.fold,
                n_folds=self.config.n_folds,
            ),
        )
        val_ds = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split="val",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=None,
                min_involvement=None,
                fold=self.config.fold,
                n_folds=self.config.n_folds,
            ),
        )
        test_ds = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split="test",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=None,
                min_involvement=None,
                fold=self.config.fold,
                n_folds=self.config.n_folds,
            ),
        )
        if self.config.debug:
            train_ds = torch.utils.data.Subset(train_ds, torch.arange(0, 100))
            test_ds = torch.utils.data.Subset(test_ds, torch.arange(0, 100))

        self.train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )

    def __call__(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.epochs):
            logging.info(f"Epoch {self.epoch}")
            self.training = True
            self.run_epoch(self.train_loader, desc="train")
            self.training = False
            val_metrics = self.run_epoch(self.val_loader, desc="val")
            tracked_metric = val_metrics["val/core_auc_high_involvement"]
            if tracked_metric > self.best_score:
                self.best_score = tracked_metric
                logging.info(f"New best score: {self.best_score}")
                metrics = self.run_epoch(self.test_loader, desc="test")
                test_score = metrics["test/core_auc_high_involvement"]
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.ckpt_dir,
                        f"best_model_epoch{self.epoch}_auc{test_score:.2f}.ckpt",
                    ),
                )

    def run_epoch(self, loader, desc="train"):
        train = self.training

        with torch.no_grad() if not train else torch.enable_grad():
            self.model.train() if train else self.model.eval()

            core_probs = []
            core_labels = []
            patch_probs = []
            patch_labels = []
            pred_involvement = []
            gt_involvement = []
            patch_involvement = []

            acc_steps = 1
            for i, batch in enumerate(tqdm(loader, desc=desc)):
                bmode, needle_mask, prostate_mask, label, involvement = batch

                bmode = bmode.cuda()
                needle_mask = needle_mask.cuda()
                prostate_mask = prostate_mask.cuda()
                label = label.cuda()

                with torch.cuda.amp.autocast(): 
                    heatmap_logits = self.model(bmode)
                    masked_prediction_module: MaskedPredictionModule = (
                        self.masked_prediction_module_train
                        if train
                        else self.masked_prediction_module_test
                    )
                    outputs: MaskedPredictionModule.Output = masked_prediction_module(
                        heatmap_logits, needle_mask, prostate_mask, label
                    )

                    loss = self.compute_loss(outputs, involvement)

                if train:
                    self.gradient_scaler.scale(loss).backward()
                    
                    if acc_steps % self.config.accumulate_grad_steps == 0:
                        self.gradient_scaler.step(self.optimizer)
                        self.gradient_scaler.update()
                        self.optimizer.zero_grad()
                        acc_steps = 1
                    else:
                        acc_steps += 1
                    self.scheduler.step()
                    wandb.log(
                        {"train_loss": loss, "lr": self.scheduler.get_last_lr()[0]}
                    )

                core_probs.append(outputs.core_predictions.detach().cpu())
                core_labels.append(outputs.core_labels.detach().cpu())
                patch_probs.append(outputs.patch_predictions.detach().cpu())
                patch_labels.append(outputs.patch_labels.detach().cpu())
                patch_predictions = (outputs.patch_predictions > 0.5).float()
                pred_involvement_batch = []
                for core_idx in outputs.core_indices.unique():
                    pred_involvement_batch.append(
                        patch_predictions[outputs.core_indices == core_idx].mean()
                    )
                pred_involvement.append(
                    torch.stack(pred_involvement_batch).detach().cpu()
                )
                patch_involvement_i = torch.zeros_like(patch_predictions)
                for core_idx in range(len(involvement)):
                    patch_involvement_i[outputs.core_indices == core_idx] = involvement[
                        core_idx
                    ]
                patch_involvement.append(patch_involvement_i.detach().cpu())
                gt_involvement.append(involvement)

                interval = 100 if train else 10
                if i % interval == 0:
                    self.show_example(batch)
                    wandb.log({f"{desc}_example": wandb.Image(plt)})
                    plt.close()

            from sklearn.metrics import roc_auc_score, balanced_accuracy_score, r2_score

            metrics = {}

            # core predictions
            core_probs = torch.cat(core_probs)
            core_labels = torch.cat(core_labels)
            metrics["core_auc"] = roc_auc_score(core_labels, core_probs)
            plt.hist(core_probs[core_labels == 0], bins=100, alpha=0.5, density=True)
            plt.hist(core_probs[core_labels == 1], bins=100, alpha=0.5, density=True)
            plt.legend(["Benign", "Cancer"])
            plt.xlabel(f"Probability of cancer")
            plt.ylabel("Density")
            plt.title(f"Core AUC: {metrics['core_auc']:.3f}")
            wandb.log(
                {
                    f"{desc}_corewise_histogram": wandb.Image(
                        plt, caption="Histogram of core predictions"
                    )
                }
            )
            plt.close()

            # involvement predictions
            pred_involvement = torch.cat(pred_involvement)
            gt_involvement = torch.cat(gt_involvement)
            metrics["involvement_r2"] = r2_score(gt_involvement, pred_involvement)
            plt.scatter(gt_involvement, pred_involvement)
            plt.xlabel("Ground truth involvement")
            plt.ylabel("Predicted involvement")
            plt.title(f"Involvement R2: {metrics['involvement_r2']:.3f}")
            wandb.log(
                {
                    f"{desc}_involvement": wandb.Image(
                        plt, caption="Ground truth vs predicted involvement"
                    )
                }
            )
            plt.close()

            # high involvement core predictions
            high_involvement = gt_involvement > 0.4
            benign = core_labels[:, 0] == 0
            keep = torch.logical_or(high_involvement, benign)
            if keep.sum() > 0:
                core_probs = core_probs[keep]
                core_labels = core_labels[keep]
                metrics["core_auc_high_involvement"] = roc_auc_score(
                    core_labels, core_probs
                )
                plt.hist(
                    core_probs[core_labels == 0], bins=100, alpha=0.5, density=True
                )
                plt.hist(
                    core_probs[core_labels == 1], bins=100, alpha=0.5, density=True
                )
                plt.legend(["Benign", "Cancer"])
                plt.xlabel(f"Probability of cancer")
                plt.ylabel("Density")
                plt.title(
                    f"Core AUC (high involvement): {metrics['core_auc_high_involvement']:.3f}"
                )
                wandb.log(
                    {
                        f"{desc}/corewise_histogram_high_involvement": wandb.Image(
                            plt, caption="Histogram of core predictions"
                        )
                    }
                )
                plt.close()

            metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
            metrics["epoch"] = self.epoch
            wandb.log(metrics)
            return metrics

    def compute_loss(self, outputs: MaskedPredictionModule.Output, involvement):
        BATCH_SIZE = outputs.core_predictions.shape[0]

        match self.config.loss:
            case "basic_ce":
                loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs.patch_logits, outputs.patch_labels
                )
            case "involvement_tolerant_loss":
                loss = torch.tensor(
                    0, dtype=torch.float32, device=outputs.core_predictions.device
                )
                for i in range(BATCH_SIZE):
                    patch_logits_for_core = outputs.patch_logits[
                        outputs.core_indices == i
                    ]
                    patch_labels_for_core = outputs.patch_labels[
                        outputs.core_indices == i
                    ]
                    involvement_for_core = involvement[i]
                    if patch_labels_for_core[0].item() == 0:
                        # core is benign, so label noise is assumed to be low
                        loss += nn.functional.binary_cross_entropy_with_logits(
                            patch_logits_for_core, patch_labels_for_core
                        )
                    elif involvement_for_core.item() > 0.65:
                        # core is high involvement, so label noise is assumed to be low
                        loss += nn.functional.binary_cross_entropy_with_logits(
                            patch_logits_for_core, patch_labels_for_core
                        )
                    else:
                        # core is of intermediate involvement, so label noise is assumed to be high.
                        # we should be tolerant of the model's "false positives" in this case.
                        pred_index_sorted_by_cancer_score = torch.argsort(
                            patch_logits_for_core[:, 0], descending=True
                        )
                        patch_logits_for_core = patch_logits_for_core[
                            pred_index_sorted_by_cancer_score
                        ]
                        patch_labels_for_core = patch_labels_for_core[
                            pred_index_sorted_by_cancer_score
                        ]
                        n_predictions = patch_logits_for_core.shape[0]
                        patch_predictions_for_core_for_loss = (
                            patch_logits_for_core[
                                : int(
                                    n_predictions * involvement_for_core.item()
                                )
                            ]
                        )
                        patch_labels_for_core_for_loss = patch_labels_for_core[
                            : int(n_predictions * involvement_for_core.item())
                        ]
                        loss += nn.functional.binary_cross_entropy_with_logits(
                            patch_predictions_for_core_for_loss,
                            patch_labels_for_core_for_loss,
                        )

            case _:
                raise ValueError(f"Unknown loss: {self.config.loss}")
            
        return loss

    @torch.no_grad()
    def show_example(self, batch):
        image, needle_mask, prostate_mask, label, involvement = batch
        image = image.cuda()
        needle_mask = needle_mask.cuda()
        prostate_mask = prostate_mask.cuda()
        label = label.cuda()

        logits = self.model(image)
        masked_prediction_module = (
            self.masked_prediction_module_train
            if self.training
            else self.masked_prediction_module_test
        )
        outputs = masked_prediction_module(logits, needle_mask, prostate_mask, label)
        # outputs: MaskedPredictionModule.Output = self.masked_prediction_module(logits, needle_mask, prostate_mask, label)
        core_prediction = outputs.core_predictions[0].item()

        pred = logits.sigmoid()

        needle_mask = needle_mask.cpu()
        prostate_mask = prostate_mask.cpu()
        logits = logits.cpu()
        pred = pred.cpu()
        image = image.cpu()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        [ax.set_axis_off() for ax in ax.flatten()]
        kwargs = dict(vmin=0, vmax=1, extent=(0, 46, 0, 28))

        ax[0].imshow(image[0].permute(1, 2, 0), **kwargs)
        prostate_mask = prostate_mask.cpu()
        ax[0].imshow(
            prostate_mask[0, 0], alpha=prostate_mask[0][0] * 0.3, cmap="Blues", **kwargs
        )
        ax[0].imshow(needle_mask[0, 0], alpha=needle_mask[0][0], cmap="Reds", **kwargs)
        ax[0].set_title(f"Ground truth label: {label[0].item()}")

        ax[1].imshow(pred[0, 0], **kwargs)

        if self.training:
            valid_loss_region = (
                needle_mask[0][0] > self.config.needle_threshold
            ).float() * (prostate_mask[0][0] > self.config.prostate_threshold).float()
        else:
            valid_loss_region = (prostate_mask[0][0] > 0.5).float() * (
                needle_mask[0][0] > 0.5
            ).float()

        alpha = torch.nn.functional.interpolate(
            valid_loss_region[None, None], size=(256, 256), mode="nearest"
        )[0, 0]
        ax[2].imshow(pred[0, 0], alpha=alpha, **kwargs)
        ax[2].set_title(f"Core prediction: {core_prediction:.3f}")

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

    def checkpoint(self):
        self.save()
        return super().checkpoint()


if __name__ == "__main__":
    Experiment.submit()
