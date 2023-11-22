import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
import logging
import wandb

import medAI
from medAI.utils.setup import BasicExperiment, BasicExperimentConfig
from baseline_experiment import BaselineExperiment, BaselineConfig, OptimizerConfig
from models.mt3_model import MT3Model, MT3Config
from utils.metrics import MetricCalculator

from timm.optim.optim_factory import create_optimizer

from einops import rearrange, repeat
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class MT3Config(BaselineConfig):
    """Configuration for the experiment."""
    exp_dir: str = "./projects/tta/logs/first_experiment_test" 
    resume: bool = False
    debug: bool = True
    use_wandb: bool = False
    
    epochs: int = 10 
    batch_size: int = 8
    fold: int = 0
    min_invovlement: int = 40
    num_support_patches: int = 10
    model_config: MT3Config = MT3Config()
    optimizer_config: OptimizerConfig = OptimizerConfig()


class MT3Experiment(BaselineExperiment): 
    config_class = MT3Config
    config: MT3Config

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
                patch = (patch - patch.min()) / (patch.max() - patch.min())
                patch = TVImage(patch)
                patch = T.ToTensor()(patch)
                patch = T.Resize(selfT.size, antialias=True)(patch).float()

                support_patches = item.pop("support_patches")
                # Normalize support patches along last two dimensions
                support_patches = (support_patches - support_patches.min(axis=(1, 2), keepdims=True)) \
                / (support_patches.max(axis=(1,2), keepdims=True) \
                    - support_patches.min(axis=(1, 2), keepdims=True))
                support_patches = TVImage(support_patches)
                support_patches = T.ToTensor()(support_patches)
                support_patches = T.Resize(selfT.size, antialias=True)(support_patches).float()
                
                # Augment support patches
                transform = T.Compose([
                    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                ])   
                support_patches_aug1, support_patches_aug2 = transform(support_patches, support_patches)
                
                if selfT.augment:
                    patch = transform(patch)
                
                label = torch.tensor(item["grade"] != "Benign").long()
                return support_patches_aug1, support_patches_aug2, patch, label, item


        from datasets.datasets import ExactNCT2013RFPatchesWithSupportPatches, CohortSelectionOptions, SupportPatchConfig
        train_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="train",
            transform=Transform(augment=False),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=1,
                min_involvement=self.config.min_invovlement,
                remove_benign_from_positive_patients=True,
                fold=self.config.fold,
            ),
            support_patch_config=SupportPatchConfig(
                num_support_patches=self.config.num_support_patches
            ),
            debug=self.config.debug,
        )
        
        val_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="val",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=None,
                min_involvement=self.config.min_invovlement,
                fold=self.config.fold
            ),
            support_patch_config=SupportPatchConfig(
                num_support_patches=self.config.num_support_patches
            ),
            debug=self.config.debug,
        )
                
        test_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="test",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=None,
                min_involvement=self.config.min_invovlement,
                fold=self.config.fold
            ),
            support_patch_config=SupportPatchConfig(
                num_support_patches=self.config.num_support_patches
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


        self.test_loaders = {
            "val": self.val_loader,
            "test": self.test_loader
        }
        
    def setup_model(self):
        model = MT3Model(self.config.model_config)
        return model
    
    def run_epoch(self, loader, train=True, desc="train"):
        with torch.no_grad() if not train else torch.enable_grad():
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
                    wandb.log({"lr": self.scheduler.get_lr()[0]})
                             
                # Update metrics   
                self.metric_calculator.update(
                    batch_meta_data = meta_data,
                    logits = logits.detach().cpu().numpy(),
                    labels = labels.detach().cpu().numpy(),
                )
                
                # Log losses
                self.log_losses(total_loss_avg, ce_loss_avg, byol_loss_avg, desc)
                
                # Break if debug
                if self.config.debug and i > 1:
                    break
            
            # Log metrics every epoch
            self.log_metrics(desc)
                
    def log_losses(self, total_loss_avg, ce_loss_avg, byol_loss_avg, desc):
        wandb.log({
            f"{desc}_loss": total_loss_avg,
            f"{desc}_ce_loss": ce_loss_avg,
            f"{desc}_byol_loss": byol_loss_avg,
            })


if __name__ == '__main__': 
    MT3Experiment.submit()