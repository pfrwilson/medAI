import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
import logging
import wandb

import medAI
from medAI.utils.setup import BasicExperiment, BasicExperimentConfig
from baseline_experiment import BaselineExperiment, BaselineConfig, OptimizerConfig
from models.mt3_model import MT3Model, MT3ANILModel, MT3Config
from utils.metrics import MetricCalculator

from timm.optim.optim_factory import create_optimizer

from einops import rearrange, repeat
from tqdm import tqdm
import matplotlib.pyplot as plt

from copy import deepcopy, copy

@dataclass
class MT3ExperimentConfig(BaselineConfig):
    """Configuration for the experiment."""
    name: str = "mt3_anil_2sprt"
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    
    epochs: int = 50
    batch_size: int = 32
    fold: int = 0
    
    min_invovlement: int = 40
    patch_size_mm: tp.Tuple[float, float] = (5, 5)
    benign_to_cancer_ratio_train: tp.Optional[float] = 1.0
    benign_to_cancer_ratio_test: tp.Optional[float] = None
    instance_norm: bool = False

    num_support_patches: int = 2
    include_query_patch: bool = False

    
    mttt_anil: bool = True
    model_config: MT3Config = MT3Config(
        inner_steps=1, 
        inner_lr=0.001,
        beta_byol=0.3,
        )


class MT3Experiment(BaselineExperiment): 
    config_class = MT3ExperimentConfig
    config: MT3ExperimentConfig

    def setup_data(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import v2 as T
        # from torchvision.datapoints import Image as TVImage
        from torchvision.tv_tensors import Image as TVImage

        class Transform:
            def __init__(selfT, augment=False):
                selfT.augment = augment
                selfT.size = (256, 256)
            
            def __call__(selfT, item):
                patch = item.pop("patch")
                patch = copy(patch)
                patch = (patch - patch.min()) / (patch.max() - patch.min()) \
                    if self.config.instance_norm else patch
                patch = TVImage(patch)
                patch = T.Resize(selfT.size, antialias=True)(patch).float()

                # Support patches
                support_patches = item.pop("support_patches")
                support_patches = copy(support_patches)
                # Normalize support patches along last two dimensions
                support_patches = (support_patches - support_patches.min(axis=(1, 2), keepdims=True)) \
                / (support_patches.max(axis=(1,2), keepdims=True) \
                    - support_patches.min(axis=(1, 2), keepdims=True)) if self.config.instance_norm else support_patches
                support_patches = TVImage(support_patches)
                support_patches = T.Resize(selfT.size, antialias=True)(support_patches).float()
                
                # Augment support patches
                transform = T.Compose([
                    T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                    T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                ])   
                support_patches_aug1, support_patches_aug2 = transform(support_patches, support_patches)
                
                if selfT.augment:
                    patch = transform(patch)
                
                label = torch.tensor(item["grade"] != "Benign").long()
                return support_patches_aug1, support_patches_aug2, patch, label, item


        from datasets.datasets import ExactNCT2013RFPatchesWithSupportPatches, CohortSelectionOptions, SupportPatchConfig, PatchOptions
        train_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="train",
            transform=Transform(augment=False),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=self.config.benign_to_cancer_ratio_train,
                min_involvement=self.config.min_invovlement,
                remove_benign_from_positive_patients=True,
                fold=self.config.fold,
            ),
            patch_options=PatchOptions(
                patch_size_mm=self.config.patch_size_mm,
                needle_mask_threshold=self.config.needle_mask_threshold,
                prostate_mask_threshold=self.config.prostate_mask_threshold,
            ),
            support_patch_config=SupportPatchConfig(
                num_support_patches=self.config.num_support_patches,
                include_query_patch=self.config.include_query_patch
            ),
            debug=self.config.debug,
        )
        
        val_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="val",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=self.config.benign_to_cancer_ratio_test,
                min_involvement=None,
                fold=self.config.fold
            ),
            patch_options=PatchOptions(
                patch_size_mm=self.config.patch_size_mm,
                needle_mask_threshold=self.config.needle_mask_threshold,
                prostate_mask_threshold=self.config.prostate_mask_threshold,
            ),
            support_patch_config=SupportPatchConfig(
                num_support_patches=self.config.num_support_patches,
                include_query_patch=self.config.include_query_patch
            ),
            debug=self.config.debug,
        )
                
        test_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="test",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=self.config.benign_to_cancer_ratio_test,
                min_involvement=None,
                fold=self.config.fold
            ),
            patch_options=PatchOptions(
                patch_size_mm=self.config.patch_size_mm,
                needle_mask_threshold=self.config.needle_mask_threshold,
                prostate_mask_threshold=self.config.prostate_mask_threshold,
            ),
            support_patch_config=SupportPatchConfig(
                num_support_patches=self.config.num_support_patches,
                include_query_patch=self.config.include_query_patch
            ),
            debug=self.config.debug,
        )


        self.train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )


        # self.test_loaders = {
        #     "val": self.val_loader,
        #     "test": self.test_loader
        # }
        
    def setup_model(self):
        if not self.config.mttt_anil:
            model = MT3Model(self.config.model_config)
        else:
            model = MT3ANILModel(self.config.model_config)
        return model
    
    def run_epoch(self, loader, train=True, desc="train"):
        # with torch.no_grad() if not train else torch.enable_grad():
        self.model.train() if train else self.model.eval()
        
        
        for i, batch in enumerate(tqdm(loader, desc=desc)):
            images_aug_1, images_aug_2, images, labels, meta_data = batch
            images_aug_1 = images_aug_1.cuda()
            images_aug_2 = images_aug_2.cuda()
            images = images.cuda()
            labels = labels.cuda()
            
            # Zero gradients
            if train:
                self.optimizer.zero_grad()
            
            (logits, 
            total_loss_avg,
            ce_loss_avg,
            byol_loss_avg
            ) = self.model(images_aug_1, images_aug_2, images, labels, training=train)

            # Optimizer step
            if train:
                # Loss already backwarded in the model
                self.optimizer.step()
                self.scheduler.step()
                wandb.log({"lr": self.scheduler.get_last_lr()[0]})
                            
            # Update metrics   
            self.metric_calculator.update(
                batch_meta_data = meta_data,
                probs = F.softmax(logits, dim=-1).detach().cpu(),
                labels = labels.detach().cpu(),
            )
            
            # Log losses
            self.log_losses(total_loss_avg, ce_loss_avg, byol_loss_avg, desc)
            
            # # Break if debug
            # if self.config.debug and i > 0:
            #     break
        
        # Log metrics every epoch
        self.log_metrics(desc)
                
    def log_losses(self, total_loss_avg, ce_loss_avg, byol_loss_avg, desc):
        wandb.log({
            f"{desc}/loss": total_loss_avg,
            f"{desc}/ce_loss": ce_loss_avg,
            f"{desc}/byol_loss": byol_loss_avg,
            "epoch": self.epoch,
            },
            commit=False
            )


if __name__ == '__main__': 
    MT3Experiment.submit()