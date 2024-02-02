import os
from cvxpy import diag, var
from dotenv import load_dotenv
from sympy import true
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
from medAI.utils.setup import BasicExperiment
from baseline_experiment import BaselineExperiment, BaselineConfig, FeatureExtractorConfig

from utils.metrics import MetricCalculator

from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
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


def diversity_loss_fn(reprs: torch.Tensor):
    """Computes the diversity loss for a set of representations.
    reprs (torch.Tensor): a tensor of shape (n_models, bz, d) containing the representations of bz samples for n_models models."""
    
    reprs = reprs.permute(1, 0, 2) # (bz, n_models, d)
    bz, n_models, d = reprs.shape
    
    # Compute the variance loss
    eps = 1e-4
    std_reprs = torch.sqrt(reprs.var(dim=1, keepdim=True) + eps) # (bz, 1, d)  
    var_loss = F.relu(1 - std_reprs).mean(-1).mean()
    
    # Compute the covariance loss
    reprs_bar = reprs - reprs.mean(dim=1, keepdim=True) # (bz, n_models, d)
    cov_reprs_bar = (reprs_bar.transpose(1, 2) @ reprs_bar) / (n_models - 1) # (bz, d, d)
    diag = torch.eye(reprs.shape[-1], device=reprs.device) # (d, d)
    diag = ~diag.bool().view(-1)
    cov_loss = (cov_reprs_bar.view(bz, -1)[:, diag].pow_(2).sum(dim=-1) / d).mean()
    # cov_loss = (cov_reprs_bar[~diag.bool()].pow_(2).sum(dim=[-1, -2]) / d).mean()
    
    return var_loss, cov_loss



@dataclass
class DivembleConfig(BaselineConfig):
    """Configuration for the experiment."""
    name: str = "Divemble_gn_5mdls_test"
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
    
    num_ensembles: int = 5
    var_reg: float = 0.5
    cov_reg: float = 0.05
    model_config: FeatureExtractorConfig = FeatureExtractorConfig(features_only=True)


class Divemblexperiment(BaselineExperiment): 
    config_class = DivembleConfig
    config: DivembleConfig

    def setup(self):
        # logging setup
        super(BaselineExperiment, self).setup()
        self.setup_data()
        self.setup_metrics()
        
        logging.info('Setting up model, optimizer, scheduler')
        self.list_fe_models, self.list_linears = self.setup_model()
        
        params = []
        for model in self.list_fe_models:
            params.append({'params': model.parameters()})
        for linear in self.list_linears:
            params.append({'params': linear.parameters()})
        
        self.optimizer = optim.Adam(
            params,
            lr=self.config.optimizer_config.lr,
            weight_decay=self.config.optimizer_config.weight_decay
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
        
        # # Testing if loading when preemption works
        # param_buffer = self._get_param_buffer_data()
        # self._set_param_buffer_data(*param_buffer)
        
        if state is not None:
            # self._set_param_buffer_data(*state["param_buffer_data"])
            [fe_model.load_state_dict(state["list_fe_models"][i])\
                for i, fe_model in enumerate(self.list_fe_models)]
            [linear.load_state_dict(state["list_linears"][i])\
                for i, linear in enumerate(self.list_linears)]
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.epoch = state["epoch"]
            self.metric_calculator.initialize_best_score(state["best_score"])
            self.best_score = state["best_score"]
            self.save_states(save_model=False) # Free up model space
        
        # Initialize best score
        self.best_score = self.metric_calculator._get_best_score_dict()
        self.best_score_updated = False

        logging.info(f"""Number of parameters: 
                     {self.config.num_ensembles*sum(p.numel() for p in self.list_fe_models[0].parameters())}""")
        logging.info(f"""Trainable parameters: 
                     {self.config.num_ensembles*sum(p.numel() for p in self.list_fe_models[0].parameters() if p.requires_grad)}""")

    def save_states(self, best_model=False, save_model=False):
        torch.save(
            {   
                "list_fe_models": [fe_model.state_dict() for fe_model in self.list_fe_models]\
                    if save_model else None,
                "list_linears": [linear.state_dict() for linear in self.list_linears]\
                    if save_model else None,
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
                    "list_fe_models": [fe_model.state_dict() for fe_model in self.list_fe_models],
                    "list_linears": [linear.state_dict() for linear in self.list_linears],
                },
                os.path.join(
                    self.ckpt_dir,
                    "best_model.ckpt",
                )
            )

    def setup_model(self):
        models = [super(Divemblexperiment, self).setup_model().cuda()
                     for _ in range(self.config.num_ensembles)]
        global_pools = [SelectAdaptivePool2d(pool_type='avg',flatten=True,input_fmt='NCHW').cuda() 
                        for _ in range(self.config.num_ensembles)]
        
        fe_models = [nn.Sequential(TimmFeatureExtractorWrapper(model), global_pool) 
                     for model, global_pool in zip(models, global_pools)]
        
        linears = [nn.Linear(512, self.config.model_config.num_classes).cuda()
                   for _ in range(self.config.num_ensembles)]
            
        return fe_models, linears

    def run_epoch(self, loader, train=True, desc="train"):
        with torch.no_grad() if not train else torch.enable_grad():
            [fe_model.train() if train else fe_model.eval() for fe_model in self.list_fe_models]
            [linear.train() if train else linear.eval() for linear in self.list_linears]
                        
            for i, batch in enumerate(tqdm(loader, desc=desc)):
                batch = deepcopy(batch)
                images, labels, meta_data = batch
                images = images.cuda()
                labels = labels.cuda()
                
                # Forward pass
                list_reprs = [model(images) for model in self.list_fe_models]
                logits = torch.stack([linear(repr) for linear, repr in zip(self.list_linears, list_reprs)])
                
                ce_losses = [nn.CrossEntropyLoss()(
                    logits[i, ...],
                    labels
                    ) for i in range(self.config.num_ensembles)
                ]
                
                var_loss, cov_loss = diversity_loss_fn(torch.stack(list_reprs))
                diversity_loss = self.config.var_reg*var_loss + self.config.cov_reg*cov_loss
                
                loss = sum(ce_losses) + diversity_loss                

                # Optimizer step
                if train:             
                    self.optimizer.zero_grad()      
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    wandb.log({"lr": self.scheduler.get_last_lr()[0]})
                             
                # Update metrics   
                self.metric_calculator.update(
                    batch_meta_data = meta_data,
                    probs = F.softmax(logits, dim=-1).mean(dim=0).detach().cpu(), # Take mean over ensembles
                    labels = labels.detach().cpu(),
                )
                
                # Log losses
                self.log_losses(loss = loss, ce_loss = sum(ce_losses), var_loss = var_loss, cov_loss = cov_loss, desc = desc)
                
                # # Break if debug
                # if self.config.debug and i > 1:
                #     break
            
            # Log metrics every epoch
            self.log_metrics(desc)
            
    def log_losses(self, loss, ce_loss, var_loss, cov_loss, desc):
        wandb.log({
            f"{desc}/total_loss": loss.item(),
            f"{desc}/loss": ce_loss.item(),
            f"{desc}/var_loss": var_loss.item(),
            f"{desc}/cov_loss": cov_loss.item(),
            "epoch": self.epoch
        })
          

class TimmFeatureExtractorWrapper(nn.Module):
    def __init__(self, timm_model):
        super(TimmFeatureExtractorWrapper, self).__init__()
        self.model = timm_model
        
    def forward(self, x):
        features = self.model(x)
        return features[-1]  # Return only the last feature map
    

if __name__ == '__main__': 
    Divemblexperiment.submit()