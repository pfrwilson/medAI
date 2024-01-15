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
from medAI.utils.setup import BasicExperiment
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
class EnsembleConfig(BaselineConfig):
    """Configuration for the experiment."""
    name: str = "ensemble_gn_5mdls"
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
    model_config: FeatureExtractorConfig = FeatureExtractorConfig()


class Ensemblexperiment(BaselineExperiment): 
    config_class = EnsembleConfig
    config: EnsembleConfig

    def setup(self):
        # logging setup
        super(BaselineExperiment, self).setup()
        self.setup_data()
        self.setup_metrics()
        
        logging.info('Setting up model, optimizer, scheduler')
        self.list_models = self.setup_model()
        
        params = []
        for model in self.list_models:
            params.append({'params': model.parameters()})
        
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
            [model.load_state_dict(state["list_models"][i])\
                for i, model in enumerate(self.list_models)]
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
                     {self.config.num_ensembles*sum(p.numel() for p in self.list_models[0].parameters())}""")
        logging.info(f"""Trainable parameters: 
                     {self.config.num_ensembles*sum(p.numel() for p in self.list_models[0].parameters() if p.requires_grad)}""")

    def save_states(self, best_model=False, save_model=False):
        torch.save(
            {   
                # "param_buffer_data": self._get_param_buffer_data() if save_model else None,
                "list_models": [model.state_dict() for model in self.list_models]\
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
                    f"list_models": [model.state_dict() for model in self.list_models],
                },
                os.path.join(
                    self.ckpt_dir,
                    f"best_model.ckpt",
                )
            )

    def setup_model(self):
        models = [super(Ensemblexperiment, self).setup_model() for _ in range(self.config.num_ensembles)]
        models = [model.cuda() for model in models]
        return models

    def run_epoch(self, loader, train=True, desc="train"):
        with torch.no_grad() if not train else torch.enable_grad():
            [model.train() if train else model.eval() for model in self.list_models]
                        
            for i, batch in enumerate(tqdm(loader, desc=desc)):
                batch = deepcopy(batch)
                images, labels, meta_data = batch
                images = images.cuda()
                labels = labels.cuda()
                               
                # Zero gradients
                if train:
                    self.optimizer.zero_grad()
                
                # Forward pass
                logits = torch.stack([model(images) for model in self.list_models])
                
                losses = [nn.CrossEntropyLoss()(
                    logits[i, ...],
                    labels
                    ) for i in range(self.config.num_ensembles)
                ]

                # Optimizer step
                if train:                   
                    sum(losses).backward()
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
                self.log_losses(sum(losses)/len(losses), desc)
                
                # # Break if debug
                # if self.config.debug and i > 1:
                #     break
            
            # Log metrics every epoch
            self.log_metrics(desc)


if __name__ == '__main__': 
    Ensemblexperiment.submit()