import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import math
import torch
import torch.nn as nn
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

from utils.sam_optimizer import SAM
from models.sar_model import SAR, configure_model, collect_params

from timm.optim.optim_factory import create_optimizer

from einops import rearrange, repeat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import timm

from copy import deepcopy
 

def marginal_entropy(outputs):
    '''Copied from https://github.com/zhangmarvin/memo/blob/main/cifar-10-exps/test_calls/test_adapt.py'''
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
batched_marginal_entropy = vmap(marginal_entropy)


@dataclass
class SARConfig:
    steps: int = 1
    episodic: bool = True
    margin_e0: float = 1.0 # default margin keeps all samples
    reset_constant_em: float = 0.01

    def __post_init__(self):
        self.margin_e0 = self.margin_e0*math.log(2)


@dataclass
class ExpConfig(BaselineConfig):
    """Configuration for the experiment."""
    name: str = "sar"
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    
    epochs: int = 50
    batch_size: int = 32
    batch_size_test: int = 32
    fold: int = 0
    
    min_invovlement: int = 40
    patch_size_mm: tp.Tuple[float, float] = (5, 5)
    benign_to_cancer_ratio_train: tp.Optional[float] = 1.0
    benign_to_cancer_ratio_test: tp.Optional[float] = None
    instance_norm: bool = False
    
    model_config: FeatureExtractorConfig = FeatureExtractorConfig()
    sar_config: SARConfig = SARConfig(steps=1)


class SARExperiment(BaselineExperiment): 
    config_class = ExpConfig
    config: ExpConfig
    
    def setup_adapt_model(self):
        base_model = configure_model(deepcopy(self.model))
        params, param_names = collect_params(base_model) # only affine params in norm layers
        # logging.info(f"Trainable test params: {param_names}")
                
        sam_optimizer = SAM(
            params,
            torch.optim.SGD,
            lr=self.config.optimizer_config.lr, # self.scheduler.get_last_lr()[0]
            rho=0.05,
            momentum=0.9
            )
                
        adapt_model = SAR(
            base_model,
            sam_optimizer,
            episodic=self.config.sar_config.episodic,
            steps=self.config.sar_config.steps,
            margin_e0=self.config.sar_config.margin_e0,
            reset_constant_em=self.config.sar_config.reset_constant_em,
            )
        
        return adapt_model

    def run_epoch(self, loader, train=True, desc="train"):      
        criterion = nn.CrossEntropyLoss()
        
        for i, batch in enumerate(tqdm(loader, desc=desc)):
            images, labels, meta_data = batch
            images = images.cuda()
            labels = labels.cuda()
            

            if not train:
                adapt_model = self.setup_adapt_model()
                logits = adapt_model(images)
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
                logits = logits.detach().cpu(),
                labels = labels.detach().cpu(),
            )
            
            # Log losses
            self.log_losses(loss, desc)
            
            # Break if debug
            if self.config.debug and i > 1:
                break
        
        # Log metrics every epoch
        self.log_metrics(desc)

    

if __name__ == '__main__': 
    SARExperiment.submit()