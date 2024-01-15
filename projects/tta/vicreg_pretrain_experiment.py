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

from models.vicreg_module import VICReg
from models.ridge_regression import RidgeRegressor
from timm.layers import create_classifier 
from models.linear_prob import LinearProb
 

@dataclass
class VicregConfig:
    sim_coeff: float = 25.0
    var_coeff: float = 25.0
    cov_coeff: float = 1.0
    proj_output_dim: int = 512
    proj_hidden_dim: int = 512    

@dataclass
class LinearProbConfig:
    linear_lr: float = 5e-3
    linear_epochs: int = 15
    

@dataclass
class PretrainConfig(BaselineConfig):
    """Configuration for the experiment."""
    name: str = " pretrain_test"
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    
    model_config: FeatureExtractorConfig = FeatureExtractorConfig(features_only=True)
    vicreg_config: VicregConfig = VicregConfig()
    linear_prob_config: LinearProbConfig = LinearProbConfig()


class VicregPretrainExperiment(BaselineExperiment): 
    config_class = PretrainConfig
    config: PretrainConfig

    def __init__(self, config: PretrainConfig):
        super().__init__(config)
        self.best_val_loss = np.inf
        self.best_score_updated = False
    
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
                    patch_augs = torch.stack([selfT.transform(patch) for _ in range(2)], dim=0)
                    return patch_augs, patch, label, item
                
                return -1, patch, label, item
        
        
        cohort_selection_options_train = copy(self.config.cohort_selection_config)
        cohort_selection_options_train.min_involvement = self.config.min_involvement_train
        cohort_selection_options_train.benign_to_cancer_ratio = self.config.benign_to_cancer_ratio_train
        cohort_selection_options_train.remove_benign_from_positive_patients = self.config.remove_benign_from_positive_patients_train
        
        train_ds = ExactNCT2013RFImagePatches(
            split="train",
            transform=Transform(augment=True),
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
            val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )

    def setup_model(self):
        from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
        
        model = super().setup_model()
        
        # Separate creation of classifier and global pool from feature extractor
        global_pool = SelectAdaptivePool2d(
            pool_type='avg',
            flatten=True,
            input_fmt='NCHW',
            ).cuda()
        
        self.model = nn.Sequential(TimmFeatureExtractorWrapper(model), global_pool)
        
        self.vicreg_model = VICReg(
            self.model,
            feature_dim=512,
            proj_output_dim=self.config.vicreg_config.proj_output_dim,
            proj_hidden_dim=self.config.vicreg_config.proj_hidden_dim,
            sim_loss_weight=self.config.vicreg_config.sim_coeff,
            var_loss_weight=self.config.vicreg_config.var_coeff,
            cov_loss_weight=self.config.vicreg_config.cov_coeff,
        )
        self.vicreg_model = self.vicreg_model.cuda()
        # note that model is still feature extractor which gets saved
        return self.model
    
    def run_epoch(self, loader, train=True, desc="train"):
        self.model.train() if train else self.model.eval()
        self.vicreg_model.train() if train else self.vicreg_model.eval()

        all_reprs_labels_metadata = []
        ssl_losses = []
        for i, batch in enumerate(tqdm(loader, desc=desc)):
            batch = deepcopy(batch)
            images_augs, images, labels, meta_data = batch
            images_augs = images_augs.cuda()
            images = images.cuda()
            labels = labels.cuda()
            
            # Forward
            ssl_loss, ssl_loss_components, r1, r2 = self.vicreg_model(images_augs[:, 0], images_augs[:, 1])
            ssl_losses.append(ssl_loss.item())
            all_reprs_labels_metadata.append((r1.detach(),labels,meta_data))            
            
            # Backward
            if train:
                self.optimizer.zero_grad()
                ssl_loss.backward()                
                self.optimizer.step()
                self.scheduler.step()
                wandb.log({"lr": self.scheduler.get_last_lr()[0]})
            
            # Log ssl losses
            self.log_losses((ssl_loss, ssl_loss_components), desc)
        
        # Linear prob train and validate
        if train:
            self.linear_prob: LinearProb = LinearProb(
                512,
                self.config.model_config.num_classes,
                ssl_epoch=self.epoch,
                metric_calculator=self.metric_calculator
                )
            self.linear_prob.train(
                all_reprs_labels_metadata,
                epochs=self.config.linear_prob_config.linear_epochs,
                lr=self.config.linear_prob_config.linear_lr
                )
        else:
            self.best_score_updated, self.best_score = self.linear_prob.validate(all_reprs_labels_metadata, desc)
        
        
        # if desc == "val":
        #     if np.mean(ssl_losses) <= self.best_val_loss:
        #         self.best_val_loss = np.mean(ssl_losses)
        #         self.best_score_updated = True
        #     else:
        #         self.best_score_updated = False
        
        '''
            # else:
            #     reprs = self.model(images)
            
            # # concat reprs and targets
            # test_reprs.append(reprs.detach())
            # test_targets.append(labels.detach())
            
            # # Break if debug
            # if self.config.debug and i > 1:
            #     break

        # # Concatenate all representations and targets for linear prob
        # test_reprs = torch.cat(test_reprs, dim=0)[None, ...]
        # test_labels = torch.cat(test_targets, dim=0)
        
        # # Linear prob + update metric calculator
        # # logits, loss = self.fit_regression_model(test_reprs, test_labels, meta_data)
        # # loss = self.linear_evaluation(test_reprs, test_labels, meta_data)
        
        # self.log_losses((loss, ssl_loss, ssl_loss_components) if train else (loss, None, None), desc)
        
        # # Log metrics every epoch
        # self.log_metrics(desc)
        '''

    def log_losses(self, losses, desc):
        ssl_loss, ssl_loss_components = losses
        wandb.log(
            {f"{desc}/ssl_loss": ssl_loss.item(),
            f"{desc}/ssl_loss_sim": ssl_loss_components[0].item(),
            f"{desc}/ssl_loss_var": ssl_loss_components[1].item(),
            f"{desc}/ssl_loss_cov": ssl_loss_components[2].item(),
            "epoch": self.epoch
            },
            commit=False
            )

    def fit_regression_model(self, test_reprs, test_labels, meta_data):
        test_targets = torch.cat([(1 - test_labels).unsqueeze(1), test_labels.unsqueeze(1)], dim=1)[None, ...]
        
        # ridge regression using test_reprs and test_targets
        linear_prob = RidgeRegressor()
        linear_prob = linear_prob.cuda()
        w, b = linear_prob(test_reprs, test_targets)
        logits = test_reprs @ w + b
        loss = nn.CrossEntropyLoss()(logits, test_targets.float())/test_targets.shape[1]
        
        # Update metrics once per epoch   
        self.metric_calculator.update(
            batch_meta_data = meta_data,
            probs = F.softmax(logits[0,...], dim=-1).detach().cpu(),
            labels = test_labels.detach().cpu(),
        )
        
        return logits, loss


class TimmFeatureExtractorWrapper(nn.Module):
    def __init__(self, timm_model):
        super(TimmFeatureExtractorWrapper, self).__init__()
        self.model = timm_model
        
    def forward(self, x):
        features = self.model(x)
        return features[-1]  # Return only the last feature map
    

if __name__ == '__main__': 
    VicregPretrainExperiment.submit()