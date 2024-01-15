import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap
import typing as tp
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
import logging
import wandb

import medAI
from baseline_experiment import BaselineExperiment, BaselineConfig, FeatureExtractorConfig

from utils.metrics import MetricCalculator

from timm.optim.optim_factory import create_optimizer

from einops import rearrange, repeat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import timm

from copy import deepcopy, copy
from simple_parsing import subgroups

from datasets.datasets import ExactNCT2013RFImagePatches
from medAI.datasets.nct2013 import (
    KFoldCohortSelectionOptions,
    LeaveOneCenterOutCohortSelectionOptions, 
    PatchOptions
)

 

def marginal_entropy(outputs):
    '''Copied from https://github.com/zhangmarvin/memo/blob/main/cifar-10-exps/test_calls/test_adapt.py'''
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
batched_marginal_entropy = vmap(marginal_entropy)

@dataclass
class MEMOConfig(BaselineConfig):
    """Configuration for the experiment."""
    name: str = "memo"
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
    
    adaptation_steps: int = 1
    adaptation_lr: float = 1e-4
    model_config: FeatureExtractorConfig = FeatureExtractorConfig()


class MEMOExperiment(BaselineExperiment): 
    config_class = MEMOConfig
    config: MEMOConfig

    def setup_data(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import v2 as T
        # from torchvision.datapoints import Image as TVImage
        from torchvision.tv_tensors import Image as TVImage

        class Transform:
            def __init__(selfT, augment=False):
                selfT.augment = augment
                selfT.size = (256, 256)
                # Augmentation
                selfT.transform = T.Compose([
                    T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                    T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                ])  
            
            def __call__(selfT, item):
                patch = item.pop("patch")
                patch = copy(patch)
                patch = (patch - patch.min()) / (patch.max() - patch.min()) \
                    if self.config.instance_norm else patch
                patch = TVImage(patch)
                patch = T.Resize(selfT.size, antialias=True)(patch).float()
                
                label = torch.tensor(item["grade"] != "Benign").long()
                
                if selfT.augment:
                    patch_augs = torch.stack([selfT.transform(patch) for _ in range(5)], dim=0)
                    return patch_augs, patch, label, item
                
                return -1, patch, label, item
        
        
        cohort_selection_options_train = copy(self.config.cohort_selection_config)
        cohort_selection_options_train.min_involvement = self.config.min_involvement_train
        cohort_selection_options_train.benign_to_cancer_ratio = self.config.benign_to_cancer_ratio_train
        cohort_selection_options_train.remove_benign_from_positive_patients = self.config.remove_benign_from_positive_patients_train
        
        train_ds = ExactNCT2013RFImagePatches(
            split="train",
            transform=Transform(augment=False),
            cohort_selection_options=cohort_selection_options_train,
            patch_options=self.config.patch_config,
            debug=self.config.debug,
        )
        
        val_ds = ExactNCT2013RFImagePatches(
            split="val",
            transform=Transform(augment=True),
            cohort_selection_options=self.config.cohort_selection_config,
            patch_options=self.config.patch_config,
            debug=self.config.debug,
        )
                
        test_ds = ExactNCT2013RFImagePatches(
            split="test",
            transform=Transform(augment=True),
            cohort_selection_options=self.config.cohort_selection_config,
            patch_options=self.config.patch_config,
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

    def run_epoch(self, loader, train=True, desc="train"):
        self.model.train() if train else self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        
        for i, batch in enumerate(tqdm(loader, desc=desc)):
            batch = deepcopy(batch)
            images_augs, images, labels, meta_data = batch
            images = images.cuda()
            labels = labels.cuda()
            

            if not train:
                batch_size, aug_size= images_augs.shape[0], images_augs.shape[1]

                # Adapt to test
                _images_augs = images_augs.reshape(-1, *images_augs.shape[2:]).cuda()
                model = deepcopy(self.model)
                model.eval()
                optimizer = optim.SGD(model.parameters(), lr=self.scheduler.get_last_lr()[0])#self.config.adaptation_lr)
                
                for j in range(self.config.adaptation_steps):
                    optimizer.zero_grad()
                    outputs = model(_images_augs).reshape(batch_size, aug_size, -1)  
                    loss, logits = batched_marginal_entropy(outputs)
                    loss.mean().backward()
                    optimizer.step()
                
                # Evaluate
                logits = self.model(images)
                loss = criterion(logits, labels)
            else:
                self.model.train()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()                
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
            self.log_losses(loss, desc)
            
            # # Break if debug
            # if self.config.debug and i > 1:
            #     break
        
        # Log metrics every epoch
        self.log_metrics(desc)

    def log_losses(self, batch_loss_avg, desc):
        wandb.log(
            {f"{desc}/loss": batch_loss_avg, "epoch": self.epoch},
            commit=False
            )

    

if __name__ == '__main__': 
    MEMOExperiment.submit()