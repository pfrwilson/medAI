from cgi import test
from email import generator
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

import torchvision
import torchvision.transforms as transforms

from timm.optim.optim_factory import create_optimizer_v2

from einops import rearrange, repeat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import timm

from copy import deepcopy, copy
from simple_parsing import subgroups

from models.vicreg_module import VICReg
from timm.layers import create_classifier 

from models.linear_prob import LinearProb


ROOT = "/scratch/ssd002/datasets"


@dataclass
class DataConfig:
    dataset_name: str = "cifar10"
    train_ratio: float = 0.8
    split_seed: int = 0

@dataclass
class FeatureExtractorConfig:
    model_name: str = 'resnet18'
    num_classes: int = 10
    in_chans: int = 3
    features_only: bool = True # return features only, not logits
    num_groups: int = 8
    use_batch_norm: bool = True
    
    def __post_init__(self):
        valid_models = timm.list_models()
        if self.model_name not in valid_models:
            raise ValueError(f"'{self.model_name}' is not a valid model. Choose from timm.list_models(): {valid_models}")
        

@dataclass
class OptimizerConfig:
    opt: str = 'adam'
    lr: float = 1e-2
    weight_decay: float = 1e-6
    # momentum: float = 0.9


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
class PretrainConfig(BasicExperimentConfig):
    """Configuration for the experiment."""
    name: str = " cifar10_pretrain_test"
    group: str = None
    project: str = "bb_ssl" 
    entity: str = "mahdigilany"
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    
    epochs: int = 100
    batch_size: int = 256
    
    data_config: DataConfig = DataConfig()
    dataset_name: str = "cifar10"
    
    optimizer_config: OptimizerConfig = OptimizerConfig()    
    model_config: FeatureExtractorConfig = FeatureExtractorConfig()
    vicreg_config: VicregConfig = VicregConfig()
    linear_prob_config: LinearProbConfig = LinearProbConfig()


class VicregPretrainExperiment(BasicExperiment): 
    config_class = PretrainConfig
    config: PretrainConfig

    def __init__(self, config: PretrainConfig):
        super().__init__(config)
        self.best_val_loss = np.inf
    
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

        logging.info('Setting up model, optimizer, scheduler')
        self.model: nn.Module = self.setup_model()
        
        self.optimizer = create_optimizer_v2(
            self.model,
            opt=self.config.optimizer_config.opt,
            lr=self.config.optimizer_config.lr,
            weight_decay=self.config.optimizer_config.weight_decay,
            )
        
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
            
        self.best_score = 0.0    
        self.best_score_updated = False

        if state is not None:
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.epoch = state["epoch"]
            self.best_score = state["best_score"]
            self.save_states(save_model=False) # Free up model space
            

        logging.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        logging.info(f"""Trainable parameters: 
                     {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}""")

    def setup_data(self):
        class Transform:
            def __init__(selfT, augment=False):
                selfT.augment = augment
                # selfT.transform = transforms.Compose([transforms.ToTensor()])
                # selfT.aug_transform = transforms.Compose([transforms.ToTensor(),
                #                       transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0), antialias=True),
                #                       transforms.RandomHorizontalFlip(p=0.5),
                #                       transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                #                       transforms.RandomGrayscale(p=0.2),
                #                       transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
                #                       ])
                selfT.transform = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
                selfT.aug_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.RandomResizedCrop(size=32, antialias=True),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                      transforms.RandomGrayscale(p=0.2),
                                      transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                                    #   transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
                                      ])
            
            def __call__(selfT, img):
                if not selfT.augment:
                    return -1, -1, selfT.transform(img)
                else:
                    return selfT.aug_transform(img), selfT.aug_transform(img), selfT.transform(img)

        if self.config.data_config.dataset_name == "cifar10":
            train_ds = torchvision.datasets.CIFAR10(root=ROOT+'/cifar10', train=True,
                                                    download=False, transform=Transform(augment=True))
            test_ds = torchvision.datasets.CIFAR10(root=ROOT+'/cifar10', train=False,
                                                    download=False, transform=Transform(augment=False))
        
        elif self.config.data_config.dataset_name == "cifar100":
            train_ds = torchvision.datasets.CIFAR100(root=ROOT+'/cifar100', train=True,
                                                    download=False, transform=Transform(augment=True))
            test_ds = torchvision.datasets.CIFAR100(root=ROOT+'/cifar100', train=False,
                                                    download=False, transform=Transform(augment=False))
        
        else:
            raise NotImplementedError
        
        # Define the sizes of your splits
        train_size = int(0.8 * len(train_ds))
        val_size = len(train_ds) - train_size

        # Split the dataset
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.data_config.split_seed)
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
    
    def setup_model(self):
        # Get number of input channels
        input_channels = self.config.model_config.in_chans
        
        # Get normalization layer
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
            in_chans=input_channels,
            features_only=self.config.model_config.features_only,
            norm_layer=norm_layer,
        ).cuda()
        
        model = self.setup_vicreg_model(model)
        
        return model
    
    def setup_vicreg_model(self, model):
        from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
        
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
            (img_aug1, img_aug2, img), labels = batch
            img_aug1 = img_aug1.cuda()
            img_aug2 = img_aug2.cuda()
            img = img.cuda()
            labels = labels.cuda()
            
            # Forward
            if desc != "test":
                ssl_loss, ssl_loss_components, r1, r2 = self.vicreg_model(img_aug1, img_aug2)
            else:
                ssl_loss, ssl_loss_components, r1, r2 = self.vicreg_model(img, img)
                
            # Log features and loss
            ssl_losses.append(ssl_loss.item())
            all_reprs_labels_metadata.append((r1.detach(), labels))            
            
            # Backward
            if train:
                self.optimizer.zero_grad()
                ssl_loss.backward()                
                self.optimizer.step()
                self.scheduler.step()
                wandb.log({"lr": self.scheduler.get_last_lr()[0]})
            
            # Log ssl losses
            self.log_losses((ssl_loss, ssl_loss_components), desc)
            
            # Debug
            if self.config.debug and i == 10:
                break
        
        # Linear prob  train and validate
        if train:
            self.linear_prob: LinearProb = LinearProb(
                512,
                self.config.model_config.num_classes,
                ssl_epoch=self.epoch,
                best_val_score=self.best_score,
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


class TimmFeatureExtractorWrapper(nn.Module):
    def __init__(self, timm_model):
        super(TimmFeatureExtractorWrapper, self).__init__()
        self.model = timm_model
        
    def forward(self, x):
        features = self.model(x)
        return features[-1]  # Return only the last feature map
    

if __name__ == '__main__': 
    VicregPretrainExperiment.submit()