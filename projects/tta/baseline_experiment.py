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
from dataclasses import dataclass, field
import logging
import wandb

import medAI
from medAI.utils.setup import BasicExperiment, BasicExperimentConfig

from utils.metrics import MetricCalculator

from timm.optim.optim_factory import create_optimizer

from einops import rearrange, repeat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import timm

from copy import copy
from simple_parsing import subgroups
from utils.sam_optimizer import SAM

from datasets.datasets import ExactNCT2013RFImagePatches
from medAI.datasets.nct2013 import (
    KFoldCohortSelectionOptions,
    LeaveOneCenterOutCohortSelectionOptions, 
    PatchOptions
)


# Avoids too many open files error from multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass 
class FourierTransformConfig:
    use_fourier_feats: bool = False
    n_fourier_feats: int = 512
    scales: tp.List[float] = field(default_factory=lambda: [0.01, 0.1, 1, 5, 10, 20, 50, 100])
    include_original: bool = False

@dataclass
class FeatureExtractorConfig:
    model_name: str = 'resnet10t'
    num_classes: int = 2
    in_chans: int = 1
    features_only: bool = False # return features only, not logits
    num_groups: int = 8
    use_batch_norm: bool = False
    
    def __post_init__(self):
        valid_models = timm.list_models()
        if self.model_name not in valid_models:
            raise ValueError(f"'{self.model_name}' is not a valid model. Choose from timm.list_models(): {valid_models}")
        

@dataclass
class OptimizerConfig:
    opt: str = 'adam'
    lr: float = 1e-4
    weight_decay: float = 0.0
    momentum: float = 0.9
    
@dataclass
class SAMOptimizerConfig:
    optimizer = torch.optim.Adam
    lr: float = 1e-4
    rho: float = 0.05
    # momentum: float = 0.9


@dataclass
class BaselineConfig(BasicExperimentConfig):
    """Configuration for the experiment."""
    # exp_dir: str = "./projects/tta/logs/first_experiment_test" 
    name: str = "baseline_gn"
    group: str = None
    project: str = "tta" 
    entity: str = "mahdigilany"
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    
    epochs: int = 50
    batch_size: int = 32
    
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
    
    model_config: FeatureExtractorConfig = FeatureExtractorConfig()
    optimizer_config: OptimizerConfig | SAMOptimizerConfig = subgroups(
        {"regular": OptimizerConfig, "sam": SAMOptimizerConfig},
        default="regular"
    )
    fourier_transform_config: FourierTransformConfig = FourierTransformConfig()
    
        
class BaselineExperiment(BasicExperiment): 
    config_class = BaselineConfig
    config: BaselineConfig

    def __call__(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.epochs):
            print(f"Epoch {self.epoch}")
            self.run_epoch(self.train_loader, train=True, desc="train")
            self.run_epoch(self.val_loader, train=False, desc="val")
            
            # Run test and save states if best score updated
            if self.best_score_updated:
                self.run_epoch(self.test_loader, train=False, desc="test")
                self.save_states(best_model=True)
            
    def setup(self):
        # logging setup
        super().setup()
        self.setup_data()
        self.setup_metrics()

        logging.info('Setting up model, optimizer, scheduler')
        self.model = self.setup_model()
        
        if isinstance(self.config.optimizer_config, SAMOptimizerConfig):
            assert isinstance(self, BaselineExperiment), "SAM only works with baseline experiment"
            self.optimizer = SAM(
                self.model.parameters(),
                self.config.optimizer_config.optimizer,
                lr=self.config.optimizer_config.lr,
                rho=self.config.optimizer_config.rho,
                # momentum=self.config.optimizer_config.momentum,
            )
        else:
            self.optimizer = create_optimizer(self.config.optimizer_config, self.model)
        
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
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.epoch = state["epoch"]
            self.metric_calculator.initialize_best_score(state["best_score"])
            self.best_score = state["best_score"]
            self.save_states(save_model=False) # Free up model space
            
        # Initialize best score if not resuming
        self.best_score = self.metric_calculator._get_best_score_dict()
            

        logging.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        logging.info(f"""Trainable parameters: 
                     {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}""")

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
                # patch = T.ToImage()(patch)
                # patch = T.ToTensor()(patch)
                patch = T.Resize(selfT.size, antialias=True)(patch).float()
                
                
                if selfT.augment:
                    # Augment support patches
                    transform = T.Compose([
                        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                        T.RandomHorizontalFlip(p=0.5),
                        T.RandomVerticalFlip(p=0.5),
                    ])  
                    patch = transform(patch)
                
                label = torch.tensor(item["grade"] != "Benign").long()
                return patch, label, item


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
            transform=Transform(),
            cohort_selection_options=self.config.cohort_selection_config,
            patch_options=self.config.patch_config,
            debug=self.config.debug,
        )
                
        test_ds = ExactNCT2013RFImagePatches(
            split="test",
            transform=Transform(),
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
        
    def setup_metrics(self):
        self.metric_calculator = MetricCalculator()
    
    def setup_model(self):
        # Create fourier transform if enabled
        if self.config.fourier_transform_config.use_fourier_feats:
            from utils.fourier_features import GaussianFourierFeatureTransform
            self.fourier_transform = GaussianFourierFeatureTransform(
                n_fourier_feats=self.config.fourier_transform_config.n_fourier_feats,
                scales=self.config.fourier_transform_config.scales,
                include_original=self.config.fourier_transform_config.include_original,
            )
        else:
            self.fourier_transform = None
        
        # Get number of input channels
        input_channels = self.config.model_config.in_chans \
            if not self.config.fourier_transform_config.use_fourier_feats \
                else self.config.fourier_transform_config.n_fourier_feats \
                        + 8*int(self.config.fourier_transform_config.include_original),
        
        if self.config.model_config.use_batch_norm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = lambda channels: nn.GroupNorm(
                    num_groups=self.config.model_config.num_groups,
                    num_channels=channels
                    )
        
        # Create the model
        model: nn.Module = timm.create_model(
            self.config.model_config.model_name,
            num_classes=self.config.model_config.num_classes,
            in_chans=input_channels[0],
            features_only=self.config.model_config.features_only,
            norm_layer=norm_layer,
        )
        
        if self.fourier_transform is not None:
            model = nn.Sequential(
                self.fourier_transform,
                model,
            )
        
        model = model.cuda()
        return model
    
    def save_states(self, best_model=False, save_model=False):
        torch.save(
            {   
                "model": self.model.state_dict() if save_model else None,
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
                    "model": self.model.state_dict(),
                    "best_score": self.best_score,
                },
                os.path.join(
                    self.ckpt_dir,
                    "best_model.ckpt",
                )
            )

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
                    
                    if isinstance(self.optimizer, SAM):
                        self.optimizer.first_step(zero_grad=True)
                        loss2 = criterion(self.model(images), labels)
                        loss2.backward()
                        loss = loss2.detach().clone()
                        self.optimizer.second_step(zero_grad=True)                 
                    else:                                       
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
        
    def log_metrics(self, desc):
        metrics = self.metric_calculator.get_metrics()
        
        # Reset metrics after each epoch
        self.metric_calculator.reset()
        
        # Update best score
        (
            best_score_updated,
            best_score
            ) = self.metric_calculator.update_best_score(metrics, desc)
        
        self.best_score_updated = copy(best_score_updated)
        self.best_score = copy(best_score)
                
        # Log metrics
        metrics_dict = {
            f"{desc}/{key}": value for key, value in metrics.items()
            }
        metrics_dict.update({"epoch": self.epoch})
        metrics_dict.update(best_score) if desc == "val" else None 
        wandb.log(
            metrics_dict,
            commit=True
            )
    
    def checkpoint(self):
        self.save_states(save_model=True)
        return super().checkpoint()



if __name__ == '__main__': 
    BaselineExperiment.submit()