import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

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
    name: str = "ensemble"
    resume: bool = False
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
    
    num_ensembles: int = 10
    model_config: FeatureExtractorConfig = FeatureExtractorConfig()


class Ensemblexperiment(BaselineExperiment): 
    config_class = EnsembleConfig
    config: EnsembleConfig
    
    def __init__(self, config: EnsembleConfig):
        super(Ensemblexperiment, self).__init__(config)
    
    def __call__(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.epochs):
            print(f"Epoch {self.epoch}")
            self.run_epoch(self.train_loader, train=True, desc="train")
            self.run_epoch(self.val_loader, train=False, desc="val")
            
            # Run test and save states if best score updated
            if any(self.list_best_score_updated):
                self.run_epoch(self.test_loader, train=False, desc="test")
                for i, if_updated in enumerate(self.list_best_score_updated):
                    if if_updated:
                        self.save_states(best_model=i)

         
    def setup(self):
        # logging setup
        super(BaselineExperiment, self).setup()
        self.setup_data()
        self.metric_calculators = [
            MetricCalculator() for _ in range(self.config.num_ensembles)
        ]
        
        logging.info('Setting up model, optimizer, scheduler')
        self.list_models = self.setup_model()
        self.model = self.setup_vmap_model()
        
        # params = []
        # for model in self.list_models:
        #     params.extend(model.parameters())
        # self.optimizer = create_optimizer(self.config.optimizer_config, params)
        
        self.optimizer = optim.Adam(
            self.params,
            lr=self.config.optimizer_config.lr,
            weight_decay=self.config.optimizer_config.weight_decay
            )
        
        self.scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=5 * len(self.train_loader),
            max_epochs=self.config.epochs * len(self.train_loader),
        )
                
        # self.optimizers = [optim.Adam(
        #     self.params[i],
        #     lr=self.config.optimizer_config.lr,
        #     weight_decay=self.config.optimizer_config.weight_decay
        #     ) for i, model in enumerate(self.list_models)
        # ]
        
        # Setup epoch and best score
        self.epoch = 0 
        self.list_best_score = [metric_calculator._get_best_score_dict()\
            for metric_calculator in self.metric_calculators]
        self.list_best_score_updated = [False for _ in range(self.config.num_ensembles)]
        
        # Load checkpoint if exists
        if "experiment.ckpt" in os.listdir(self.ckpt_dir) and self.config.resume:
            state = torch.load(os.path.join(self.ckpt_dir, "experiment.ckpt"))
            logging.info(f"Resuming from epoch {state['epoch']}")
        else:
            state = None
            
        if state is not None:
            [model.load_state_dict(state["list_models"][i])\
                for i, model in enumerate(self.list_models)]
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.epoch = state["epoch"]
            [metric_calculator.initialize_best_score(state["list_best_score"][i])\
                for i, metric_calculator in enumerate(self.metric_calculators)]
            self.save_states(save_model=False) # Free up model space
            

        logging.info(f"""Number of parameters: 
                     {self.config.num_ensembles*sum(p.numel() for p in self.list_models[0].parameters())}""")
        logging.info(f"""Trainable parameters: 
                     {self.config.num_ensembles*sum(p.numel() for p in self.list_models[0].parameters() if p.requires_grad)}""")

    def save_states(self, best_model=None, save_model=False):
        torch.save(
            {   
                "list_models": [self.list_models[i].state_dict()\
                    for i in range(self.config.num_ensembles)]\
                        if save_model else None,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "list_best_score": self.list_best_score,
            },
            os.path.join(
                self.ckpt_dir,
                "experiment.ckpt",
            )
        )
        if best_model:
            torch.save(
                {   
                    f"model_{best_model}": self.list_models[best_model].state_dict(),
                    f"best_score_{best_model}": self.best_scores[best_model],
                },
                os.path.join(
                    self.ckpt_dir,
                    f"best_model_{best_model}.ckpt",
                )
            )

    def setup_model(self):
        models = [super(Ensemblexperiment, self).setup_model() for _ in range(self.config.num_ensembles)]
        return models
    
    def setup_vmap_model(self):
        # from torch.func import stack_module_state
        # from torch.func import functional_call
        from functorch import combine_state_for_ensemble
        
        # Stack model state
        # self.params, self.buffers = stack_module_state(self.list_models)
        fmodel, self.params, self.buffers = combine_state_for_ensemble(self.list_models)
        [p.requires_grad_() for p in self.params]

        # # Construct a "stateless" version of one of the models. It is "stateless" in
        # # the sense that the parameters are meta Tensors and do not have storage.
        # base_model = deepcopy(self.list_models[0])
        # base_model = base_model.to('meta')
        
        # def fmodel(params, buffers, x):
        #     return functional_call(base_model, (params, buffers), (x,))
        
        return vmap(fmodel, in_dims=(0, 0, None))
    
    def run_epoch(self, loader, train=True, desc="train"):
        with torch.no_grad() if not train else torch.enable_grad():
            [model.train() if train else model.eval() for model in self.list_models]
            
            criterion = nn.CrossEntropyLoss()
            
            for i, batch in enumerate(tqdm(loader, desc=desc)):
                images, labels, meta_data = batch
                images = images.cuda()
                labels = labels.cuda()
                
                batch_size = images.shape[0]
                
                # Zero gradients
                if train:
                    self.optimizer.zero_grad()
                
                logits = self.model(self.params, self.buffers, images)
                
                # loss = criterion(
                #     logits.reshape(self.config.num_ensembles*batch_size, -1),
                #     labels.unsqueeze(0).repeat(logits.shape[0], 1).reshape(-1)
                #     )
                
                losses = [criterion(
                    logits[i, ...],
                    labels
                    ) for i in range(self.config.num_ensembles)
                ]

                
                # Optimizer step
                if train:
                    [loss.backward() for loss in losses]
                    # loss.backward()                                    
                    self.optimizer.step()
                    # [optimizer.ste p() for optimizer in self.optimizers]
                    self.scheduler.step()
                    wandb.log({"lr": self.scheduler.get_last_lr()[0]})
                             
                # Update metrics   
                [metric_calculator.update(
                    batch_meta_data = meta_data,
                    logits = logits[i, ...].detach().cpu(), # Take mean over ensembles
                    labels = labels.detach().cpu(),
                ) for i, metric_calculator in enumerate(self.metric_calculators)]
                
                # Log losses
                self.log_losses(sum(losses)/len(losses), desc)
                
                # # Break if debug
                # if self.config.debug and i > 1:
                #     break
            
            # Log metrics every epoch
            self.log_metrics(desc)

    def log_metrics(self, desc):
        # Get list of metrics for each model
        list_metrics = [metric_calculator.get_metrics() \
            for metric_calculator in self.metric_calculators]
        
        # Reset metrics for each model after each epoch
        [metric_calculator.reset() for metric_calculator in self.metric_calculators]
        
        # Update best scores and get best scores for each model
        for i, metrics in enumerate(list_metrics):
            best_score_updated, best_score = self.metric_calculators[i].update_best_score(metrics, desc)
            if (desc=='val' and best_score_updated) or (desc=='test' and self.list_best_score_updated[i]):
                self.list_best_score[i] = copy(best_score)
            if desc=='val':
                self.list_best_score_updated[i] = copy(best_score_updated)

        average_best_scores_dict = {}
        for key in self.list_best_score[0].keys():
            average_best_scores_dict[key] = sum([best_score[key] for best_score in self.list_best_score])/self.config.num_ensembles

        # Log metrics
        average_metrics_dict = {}
        for key in list_metrics[0].keys():
            average_metrics_dict[f"{desc}/{key}"] = sum([metrics[key] for metrics in list_metrics])/self.config.num_ensembles
            
        average_metrics_dict.update({"epoch": self.epoch})
        average_metrics_dict.update(average_best_scores_dict) if desc == "val" else None 
        wandb.log(
            average_metrics_dict,
            commit=True
            )
    

if __name__ == '__main__': 
    Ensemblexperiment.submit()