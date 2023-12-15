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
from medsam_cancer_detection_v2_model_registry import model_registry, CancerDetectorOutput
from typing import Any, Literal


@dataclass
class Config(BasicExperimentConfig, Serializable):
    """Training configuration"""

    project: str = "medsam_cancer_detection_v2"
    fold: int = 0
    benign_cancer_ratio_for_training: float | None = None
    epochs: int = 10
    optimizer: str = choice(
        "adam", "sgdw", default="adam"
    )
    use_augmentation: bool = False
    lr: float = 0.00001
    wd: float = 0.0
    batch_size: int = 64
    debug: bool = False
    accumulate_grad_steps: int = 1
    min_involvement_pct_training: float = 40.
    mode: Literal["bmode", "rf"] = "bmode"


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
        from trusnet.modeling.registry import resnet10
        self.model = resnet10().cuda()

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

        logging.info("Number of parameters: ", sum(p.numel() for p in self.model.parameters()))
        logging.info(
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
                patch = item["patch"]
                patch = T.ToTensor()(patch)
                patch = T.Resize((256, 256), antialias=True)(patch)
                patch = (patch - patch.min()) / (patch.max() - patch.min())
                patch = Image(patch).float()

                if self.augment:
                    patch = T.RandomAffine(
                        degrees=0, translate=(0.2, 0.2)
                    )(patch)

                label = torch.tensor(item["grade"] != "Benign").long()
                pct_cancer = item["pct_cancer"]
                if np.isnan(pct_cancer):
                    pct_cancer = 0

                involvement = torch.tensor(pct_cancer / 100).float()
                core_id = torch.tensor(item["id"]).long()

                return patch, label, involvement, core_id

        from medAI.datasets import ExactNCT2013BModeImages, CohortSelectionOptions, ExactNCT2013BmodePatches, PatchOptions, ExactNCT2013RFImagePatches

        patch_options = PatchOptions(
            needle_mask_threshold=0.6, prostate_mask_threshold=0.9
        )
        match self.config.mode:
            case "bmode":
                train_ds = ExactNCT2013BmodePatches(
                    split="train",
                    transform=Transform(augment=self.config.use_augmentation),
                    patch_options=patch_options,
                    cohort_selection_options=CohortSelectionOptions(
                        benign_to_cancer_ratio=self.config.benign_cancer_ratio_for_training,
                        min_involvement=self.config.min_involvement_pct_training,
                        remove_benign_from_positive_patients=True,
                        fold=self.config.fold,
                    ),
                )
                test_ds = ExactNCT2013BmodePatches(
                    split="test",
                    transform=Transform(),
                    patch_options=patch_options,
                    cohort_selection_options=CohortSelectionOptions(
                        benign_to_cancer_ratio=None, min_involvement=None, fold=self.config.fold
                    ),
                )
            case "rf":
                train_ds = ExactNCT2013RFImagePatches(
                    split="train",
                    transform=Transform(augment=self.config.use_augmentation),
                    patch_options=patch_options,
                    cohort_selection_options=CohortSelectionOptions(
                        benign_to_cancer_ratio=self.config.benign_cancer_ratio_for_training,
                        min_involvement=self.config.min_involvement_pct_training,
                        remove_benign_from_positive_patients=True,
                        fold=self.config.fold,
                    ),
                )
                test_ds = ExactNCT2013RFImagePatches(
                    split="test",
                    transform=Transform(),
                    patch_options=patch_options,
                    cohort_selection_options=CohortSelectionOptions(
                        benign_to_cancer_ratio=None, min_involvement=None, fold=self.config.fold
                    ),
                )
        
        if self.config.debug:
            train_ds = torch.utils.data.Subset(train_ds, np.random.choice(len(train_ds), self.config.batch_size * 100))
            test_ds = torch.utils.data.Subset(test_ds, np.random.choice(len(test_ds), self.config.batch_size * 100))
        
        self.train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )

    def __call__(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.epochs):
            logging.info(f"Epoch {self.epoch}")
            self.run_epoch(self.train_loader, train=True, desc="train")
            self.run_epoch(self.test_loader, train=False, desc="test")

    def run_epoch(self, loader, train=True, desc="train"):
        with torch.no_grad() if not train else torch.enable_grad():
            self.model.train() if train else self.model.eval()

            patch_probs = []
            patch_labels = []
            core_ids = [] 
            gt_involvement = []

            acc_steps = 1
            for i, batch in enumerate(tqdm(loader, desc=desc)):
                patch, label, involvement, core_id = batch
                
                patch = patch.cuda()
                label = label.cuda() 
                involvement = involvement.cuda()
                core_id = core_id.cuda()

                patch_logits = self.model(patch)
                loss = torch.nn.functional.cross_entropy(patch_logits, label)
                
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

                patch_probs.append(torch.softmax(patch_logits, dim=-1)[:, 1].detach().cpu())
                patch_labels.append(label.detach().cpu())
                core_ids.append(core_id.detach().cpu())
                gt_involvement.append(involvement.detach().cpu())

            patch_probs = torch.cat(patch_probs)
            patch_labels = torch.cat(patch_labels)
            core_ids = torch.cat(core_ids)
            gt_involvement = torch.cat(gt_involvement)

            # compute corewise predictions and labels
            core_probs = []
            core_pred_inv = []
            core_gt_inv = []
            core_labels = []
            for core_id in torch.unique(core_ids):
                keep = core_ids == core_id
                patch_probs_for_core = patch_probs[keep]
                core_probs.append(patch_probs_for_core.mean())
                pred_involvement = (patch_probs_for_core > 0.5).float().mean()
                core_pred_inv.append(pred_involvement)
                gt_involvement_for_core = gt_involvement[keep][0]
                core_gt_inv.append(gt_involvement_for_core)
                core_labels.append(patch_labels[keep][0])
            core_probs = torch.stack(core_probs)
            core_pred_inv = torch.stack(core_pred_inv)
            core_gt_inv = torch.stack(core_gt_inv)
            core_labels = torch.stack(core_labels)

            from sklearn.metrics import roc_auc_score, balanced_accuracy_score, r2_score
            metrics = {}
            
            # core predictions
            metrics["core_auc"] = roc_auc_score(core_labels, core_probs)
            plt.hist(core_probs[core_labels == 0], bins=100, alpha=0.5, density=True)
            plt.hist(core_probs[core_labels == 1], bins=100, alpha=0.5, density=True)
            plt.legend(["Benign", "Cancer"])
            plt.xlabel(f"Probability of cancer")
            plt.ylabel("Density")
            plt.title(f"Core AUC: {metrics['core_auc']:.3f}")
            wandb.log({f"{desc}_corewise_histogram": wandb.Image(plt, caption="Histogram of core predictions")})
            plt.close()

            # involvement predictions           
            pred_involvement = core_pred_inv
            metrics['involvement_r2'] = r2_score(core_gt_inv, pred_involvement)
            plt.scatter(core_gt_inv, pred_involvement)
            plt.xlabel("Ground truth involvement")
            plt.ylabel("Predicted involvement")
            plt.title(f"Involvement R2: {metrics['involvement_r2']:.3f}")
            wandb.log({f"{desc}_involvement": wandb.Image(plt, caption="Ground truth vs predicted involvement")})
            plt.close()

            # high involvement core predictions
            high_involvement = core_gt_inv > 0.4
            benign = core_labels == 0
            keep = torch.logical_or(high_involvement, benign)
            if keep.sum() > 0: 
                core_probs = core_probs[keep]
                core_labels = core_labels[keep]
                metrics["core_auc_high_involvement"] = roc_auc_score(core_labels, core_probs)
                plt.hist(core_probs[core_labels == 0], bins=100, alpha=0.5, density=True)
                plt.hist(core_probs[core_labels == 1], bins=100, alpha=0.5, density=True)
                plt.legend(["Benign", "Cancer"])
                plt.xlabel(f"Probability of cancer")
                plt.ylabel("Density")
                plt.title(f"Core AUC (high involvement): {metrics['core_auc_high_involvement']:.3f}")
                wandb.log({f"{desc}_corewise_histogram_high_involvement": wandb.Image(plt, caption="Histogram of core predictions")})
                plt.close()

            # high involvement patch predictions
            high_involvement = gt_involvement > 0.4
            benign = patch_labels == 0
            keep = torch.logical_or(high_involvement, benign)
            if keep.sum() > 0:
                patch_probs = patch_probs[keep]
                patch_labels = patch_labels[keep]
                metrics["patch_auc"] = roc_auc_score(patch_labels, patch_probs)
                plt.hist(patch_probs[patch_labels == 0], bins=100, alpha=0.5, density=True)
                plt.hist(patch_probs[patch_labels == 1], bins=100, alpha=0.5, density=True)
                plt.legend(["Benign", "Cancer"])
                plt.xlabel("Probability of cancer")
                plt.ylabel("Density")
                plt.title(f"Patch AUC: {metrics['patch_auc']:.3f}")
                wandb.log({f"{desc}_patchwise_histogram": wandb.Image(plt, caption="Histogram of patch predictions")})
                plt.close()
            
            metrics = {f"{desc}_{k}": v for k, v in metrics.items()}
            metrics["epoch"] = self.epoch
            wandb.log(metrics)
            
    def checkpoint(self):
        self.save()
        return super().checkpoint()



if __name__ == "__main__":
    Experiment.submit()
