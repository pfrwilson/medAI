import os
from turtle import back
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
from baseline_experiment import BaselineExperiment, BaselineConfig, FeatureExtractorConfig

from utils.metrics import MetricCalculator, CoreMetricCalculator

from timm.optim.optim_factory import create_optimizer

from einops import rearrange, repeat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import timm

from copy import deepcopy, copy
from simple_parsing import subgroups

from datasets.datasets import ExactNCT2013RFImagePatches, ExactNCT2013RFCores
from medAI.datasets.nct2013 import (
    KFoldCohortSelectionOptions,
    LeaveOneCenterOutCohortSelectionOptions, 
    PatchOptions
)

from models.vicreg_module import VICReg
from models.ridge_regression import RidgeRegressor
from timm.layers import create_classifier 
from models.linear_prob import LinearProb
from models.attention import MultiheadAttention as SimpleMultiheadAttention
 
 
# # Avoids too many open files error from multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class FinetunerConfig:
    train_backbone: bool = True
    backbone_lr: float = 1e-3
    head_lr: float = 1e-3
    checkpoint_path_name: str = None 
    

@dataclass
class FinetuneConfig(BaselineConfig):
    """Configuration for the experiment."""
    name: str = " finetune_test_5"
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    
    epochs: int = 100
    cohort_selection_config: KFoldCohortSelectionOptions | LeaveOneCenterOutCohortSelectionOptions = subgroups(
        {"kfold": KFoldCohortSelectionOptions(fold=0), "loco": LeaveOneCenterOutCohortSelectionOptions(leave_out='JH')},
        default="loco"
    )
    
    model_config: FeatureExtractorConfig = FeatureExtractorConfig(features_only=True)
    finetuner_config: FinetunerConfig = FinetunerConfig()


class FinetuneExperiment(BaselineExperiment): 
    config_class = FinetuneConfig
    config: FinetuneConfig

    def __init__(self, config: FinetuneConfig):
        super().__init__(config)
        self.best_val_loss = np.inf
        self.best_score_updated = False
        assert self.config.finetuner_config.checkpoint_path_name is not None, "Please provide a checkpoint path name for loading the pre-trained model"
        self._checkpoint_path = os.path.join(
            os.getcwd(),
            # f'projects/tta/logs/tta/vicreg_pretrn_2048zdim_gn_loco2/vicreg_pretrn_2048zdim_gn_loco2_{self.config.cohort_selection_config.leave_out}/', 
            f'logs/tta/{self.config.finetuner_config.checkpoint_path_name}/{self.config.finetuner_config.checkpoint_path_name}_{self.config.cohort_selection_config.leave_out}/', 
            'best_model.ckpt'
            )
        
    def setup(self):
        # logging setup
        super(BaselineExperiment, self).setup()
        self.setup_data()
        self.setup_metrics()

        logging.info('Setting up model, optimizer, scheduler')
        self.fe_model, self.linear = self.setup_model()
        
        # Optimizer
        if self.config.finetuner_config.train_backbone:
            params = [
                {"params": self.fe_model.parameters(), "lr": self.config.finetuner_config.backbone_lr},
                {"params": self.linear.parameters(), "lr": self.config.finetuner_config.head_lr}
                ]
        else:
            params = [
                {"params": self.linear.parameters(),  "lr": self.config.finetuner_config.head_lr}
                ]
        
        self.optimizer = optim.Adam(params, weight_decay=1e-6) #lr=self.config.finetuner_config.backbone_lr,
        self.scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=5 * len(self.train_loader),
            max_epochs=self.config.epochs * len(self.train_loader),
        )
        
        # Setup epoch and best score
        self.epoch = 0 
        
        # Load checkpoint if exists
        if "experiment.ckpt" in os.listdir(self.ckpt_dir) and self.config.resume:
            state = torch.load(os.path.join(self.ckpt_dir, "experiment.ckpt"))
            logging.info(f"Resuming from epoch {state['epoch']}")
        else:
            state = None
            
        if state is not None:
            self.fe_model.load_state_dict(state["fe_model"])
            self.linear.load_state_dict(state["linear"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.epoch = state["epoch"]
            self.metric_calculator.initialize_best_score(state["best_score"])
            self.best_score = state["best_score"]
            self.save_states(save_model=False) # Free up model space
            
        # Initialize best score if not resuming
        self.best_score = self.metric_calculator._get_best_score_dict()
            

        logging.info(f"Number of fe parameters: {sum(p.numel() for p in self.fe_model.parameters())}")
        logging.info(f"Number of linear parameters: {sum(p.numel() for p in self.linear.parameters())}")

    def setup_data(self):
        from torchvision.transforms import v2 as T
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
            transform=Transform(augment=False),
            cohort_selection_options=cohort_selection_options_train,
            patch_options=self.config.patch_config,
            debug=self.config.debug,
        )

        val_ds = ExactNCT2013RFImagePatches(
            split="val",
            transform=Transform(augment=False),
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
            train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4 #, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4 #, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4 #, pin_memory=True
        )

    def setup_model(self):
        from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
        
        fe_model = super().setup_model()
        global_pool = SelectAdaptivePool2d(
            pool_type='avg',
            flatten=True,
            input_fmt='NCHW',
            ).cuda()
        fe_model = nn.Sequential(TimmFeatureExtractorWrapper(fe_model), global_pool)
        fe_model.load_state_dict(torch.load(self._checkpoint_path)["model"])
        
        linear = nn.Linear(512, self.config.model_config.num_classes)        

        return fe_model.cuda(), linear.cuda()
    
    def save_states(self, best_model=False, save_model=False):
        torch.save(
            {   
                "fe_model": self.fe_model.state_dict() if save_model else None,
                "linear": self.linear.state_dict() if save_model else None,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
            },
            os.path.join(
                self.ckpt_dir,
                "experiment.ckpt",
            )
        )
        if best_model:
            torch.save(
                {   
                    "fe_model": self.fe_model.state_dict(),
                    "linear": self.linear.state_dict(),
                    "best_score": self.best_score,
                },
                os.path.join(
                    self.ckpt_dir,
                    "best_model.ckpt",
                )
            )

    def run_epoch(self, loader, train=True, desc="train"):
        self.fe_model.train() if train else self.fe_model.eval()
        self.linear.train() if train else self.linear.eval()
        
        self.metric_calculator.reset()
        
        for batch in tqdm(loader, desc=desc):
            images_augs, images, labels, meta_data = batch
            # images_augs = images_augs.cuda()
            images = images.cuda()
            labels = labels.cuda()
            
            reprs = self.fe_model(images)
            logits = self.linear(reprs)
            loss = nn.CrossEntropyLoss()(logits, labels)
                
            if desc == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                learning_rates = {f"lr_group_{i}": lr for i, lr in enumerate(self.scheduler.get_last_lr())}
                # wandb.log({"lr": self.scheduler.get_last_lr()[0]})
                wandb.log(learning_rates)
            
            self.log_losses(loss.item(), desc)
            
            # Update metrics   
            self.metric_calculator.update(
                batch_meta_data = meta_data,
                probs = F.softmax(logits, dim=-1).detach().cpu(),
                labels = labels.detach().cpu(),
            )
        
        # Log metrics every epoch
        self.log_metrics(desc)


class TimmFeatureExtractorWrapper(nn.Module):
    def __init__(self, timm_model):
        super(TimmFeatureExtractorWrapper, self).__init__()
        self.model = timm_model
        
    def forward(self, x):
        features = self.model(x)
        return features[-1]  # Return only the last feature map
    

if __name__ == '__main__': 
    FinetuneExperiment.submit()