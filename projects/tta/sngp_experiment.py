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



@dataclass
class SNGPConfig(BaselineConfig):
    """Configuration for the experiment."""
    name: str = "sngp"
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


class SNGPExperiment(BaselineExperiment): 
    config_class = SNGPConfig
    config: SNGPConfig

    def setup_model(self):      
        from models.sngp.spectral_resnets import spectral_resnet10, spectral_resnet_feature_extractor
        from models.sngp.gp_approx_models import Laplace
        
        num_deep_features=512
        num_gp_features=128
        mean_field_factor=25
        
        spectral_resnet = spectral_resnet10()
        spectral_resnet_fe = spectral_resnet_feature_extractor(spectral_resnet)
        
        sngp_model = Laplace(
            spectral_resnet_fe,
            num_deep_features=num_deep_features,
            num_gp_features=num_gp_features,
            mean_field_factor=mean_field_factor,
            num_data=len(self.train_dataset),
        )
        
        return sngp_model

    def run_epoch(self, loader, train=True, desc="train"):
        with torch.no_grad() if not train else torch.enable_grad():
            self.model.train() if train else self.model.eval()
            
            criterion = nn.CrossEntropyLoss()
            
            for i, batch in enumerate(tqdm(loader, desc=desc)):
                images, labels, meta_data = batch
                images = images.cuda()
                labels = labels.cuda()
                
                # Zero gradients
                if train:
                    self.optimizer.zero_grad()
                
                logits = self.model(images)
                
                loss = criterion(logits, labels)
                
                # Optimizer step
                if train:
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


if __name__ == '__main__': 
    SNGPExperiment.submit()