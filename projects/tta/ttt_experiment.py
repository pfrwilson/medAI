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
from models.ttt_model import TTTModel, TTTConfig
from utils.metrics import MetricCalculator

from timm.optim.optim_factory import create_optimizer

from einops import rearrange, repeat
from tqdm import tqdm
import matplotlib.pyplot as plt

from copy import deepcopy, copy
from simple_parsing import subgroups

from datasets.datasets import SupportPatchConfig, ExactNCT2013RFPatchesWithSupportPatches
from medAI.datasets.nct2013 import (
    KFoldCohortSelectionOptions,
    LeaveOneCenterOutCohortSelectionOptions, 
    PatchOptions
)


@dataclass
class TTTExperimentConfig(BaselineConfig):
    """Configuration for the experiment."""
    name: str = "ttt_2sprt"
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    
    epochs: int = 50
    batch_size: int = 32
    batch_size_test: int = 32
    shffl_test: bool = False
    
    instance_norm: bool = False
    min_involvement_train: int = 40.
    benign_to_cancer_ratio_train: tp.Optional[float] = 1.0
    remove_benign_from_positive_patients_train: bool = True
    patch_config: PatchOptions = PatchOptions(
        needle_mask_threshold = 0.6,
        prostate_mask_threshold = 0.9,
        patch_size_mm = (5, 5)
    )
    cohort_selection_config: KFoldCohortSelectionOptions | LeaveOneCenterOutCohortSelectionOptions = subgroups(
        {"kfold": KFoldCohortSelectionOptions(fold=0), "loco": LeaveOneCenterOutCohortSelectionOptions(leave_out='JH')},
        default="kfold"
    )
    
    num_support_patches: int = 2
    include_query_patch: bool = False
    
    model_config: TTTConfig = TTTConfig(
        adaptation_steps=1,
        beta_byol=0.3,
        joint_training=False,
        adaptation_lr=1e-4,
        )


class TTTExperiment(BaselineExperiment): 
    config_class = TTTExperimentConfig
    config: TTTExperimentConfig

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


        cohort_selection_options_train = copy(self.config.cohort_selection_config)
        cohort_selection_options_train.min_involvement = self.config.min_involvement_train
        cohort_selection_options_train.benign_to_cancer_ratio = self.config.benign_to_cancer_ratio_train
        cohort_selection_options_train.remove_benign_from_positive_patients = self.config.remove_benign_from_positive_patients_train
        
                
        train_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="train",
            transform=Transform(augment=False),
            cohort_selection_options=cohort_selection_options_train,
            patch_options=self.config.patch_config,
            support_patch_config=SupportPatchConfig(
                num_support_patches=self.config.num_support_patches,
                include_query_patch=self.config.include_query_patch
            ),
            debug=self.config.debug,
        )
        
        val_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="val",
            transform=Transform(),
            cohort_selection_options=self.config.cohort_selection_config,
            patch_options=self.config.patch_config,
            support_patch_config=SupportPatchConfig(
                num_support_patches=self.config.num_support_patches,
                include_query_patch=self.config.include_query_patch
            ),
            debug=self.config.debug,
        )
        
        
        if isinstance(self.config.cohort_selection_config, LeaveOneCenterOutCohortSelectionOptions):
            if self.config.cohort_selection_config.leave_out == "UVA":
                self.config.cohort_selection_config.benign_to_cancer_ratio = 5.0 
                
        test_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="test",
            transform=Transform(),
            cohort_selection_options=self.config.cohort_selection_config,
            patch_options=self.config.patch_config,
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
            val_ds, batch_size=self.config.batch_size_test, shuffle=self.config.shffl_test, num_workers=4
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size_test, shuffle=self.config.shffl_test, num_workers=4
        )
        
    def setup_model(self):
        model = TTTModel(self.config.model_config)
        model = model.cuda()
        return model
    
    def run_epoch(self, loader, train=True, desc="train"):
        self.model.train() if train else self.model.eval()
        
        for i, batch in enumerate(tqdm(loader, desc=desc)):
            batch = deepcopy(batch)
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
    TTTExperiment.submit()