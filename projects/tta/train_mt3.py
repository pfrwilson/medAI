import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
import logging
import wandb

import medAI
from medAI.utils.setup import BasicExperiment, BasicExperimentConfig

from models.model import MT3Model, MT3Config
from utils.metrics import MetricCalculator

from timm.optim.optim_factory import create_optimizer

from einops import rearrange, repeat
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class OptimizerConfig:
    opt: str = 'adam'
    lr: float = 1e-3
    weight_decay: float = 0.0
        

@dataclass
class Config(BasicExperimentConfig):
    """Configuration for the experiment."""
    exp_dir: str = "logs/first_experiment_test" 
    group: str = None
    project: str = "MT3"
    entity: str = "mahdigilany"
    resume: bool = False
    debug: bool = True
    use_wandb: bool = False
    
    epochs: int = 10 
    batch_size: int = 8
    fold: int = 0
    min_invovlement: int = 40
    num_support_patches: int = 10
    model_config: MT3Config = MT3Config()
    optimizer_config: OptimizerConfig = OptimizerConfig()


class MT3Experiment(BasicExperiment): 
    config_class = Config
    config: Config

    def __call__(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.epochs):
            print(f"Epoch {self.epoch}")
            self.run_epoch(self.train_loader, train=True, desc="train")
            for name, test_loader in self.test_loaders.items():
                self.run_epoch(self.test_loader, train=False, desc=name)
            
            if self.best_score_updated:
                self.save_states()
            
    def setup(self):
        # logging setup
        super().setup()
        self.setup_data()
        self.setup_metrics()        

        if "experiment.ckpt" in os.listdir(self.ckpt_dir):
            state = torch.load(os.path.join(self.ckpt_dir, "experiment.ckpt"))
        else:
            state = None

        logging.info('Setting up model, optimizer, scheduler')
        self.model = MT3Model(mt3_config=self.config.model_config)
        self.model = self.model.cuda()
        
        self.optimizer = create_optimizer(self.config.optimizer_config, self.model)
        self.scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=5 * len(self.train_loader),
            max_epochs=self.config.epochs * len(self.train_loader),
        )
        
        if state is not None:
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])

        logging.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        logging.info(f"""Trainable parameters: 
                     {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}""")

        self.epoch = 0 if state is None else state["epoch"]
        self.best_score = 0 if state is None else state["best_score"]

    def setup_data(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import v2 as T
        from torchvision.datapoints import Image as TVImage

        class Transform:
            def __init__(selfT, augment=False):
                selfT.augment = augment
                selfT.size = (256, 256)
            
            def __call__(selfT, item):
                patch = item.pop("patch")
                patch = (patch - patch.min()) / (patch.max() - patch.min())
                patch = TVImage(patch)
                patch = T.ToTensor()(patch)
                patch = T.Resize(selfT.size, antialias=True)(patch)

                support_patches = item.pop("support_patches")
                # Normalize support patches along last two dimensions
                support_patches = (support_patches - support_patches.min(axis=(1, 2), keepdims=True)) \
                / (support_patches.max(axis=(1,2), keepdims=True) \
                    - support_patches.min(axis=(1, 2), keepdims=True))
                support_patches = TVImage(support_patches)
                support_patches = T.ToTensor()(support_patches)
                support_patches = T.Resize(selfT.size, antialias=True)(support_patches)
                
                # Augment support patches
                transform = T.Compose([
                    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                ])   
                support_patches_aug1, support_patches_aug2 = transform(support_patches, support_patches)
                
                if selfT.augment:
                    patch = transform(patch)
                
                label = torch.tensor(item["grade"] != "Benign").long()
                return support_patches_aug1, support_patches_aug2, patch, label, item


        from datasets.datasets import ExactNCT2013RFPatchesWithSupportPatches, CohortSelectionOptions, SupportPatchConfig
        train_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="train",
            transform=Transform(augment=False),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=1,
                min_involvement=self.config.min_invovlement,
                remove_benign_from_positive_patients=True,
                fold=self.config.fold,
            ),
            support_patch_config=SupportPatchConfig(
                num_support_patches=self.config.num_support_patches
            ),
            debug=self.config.debug,
        )
        
        val_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="val",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=None,
                min_involvement=self.config.min_invovlement,
                fold=self.config.fold
            ),
            debug=self.config.debug,
        )
                
        test_ds = ExactNCT2013RFPatchesWithSupportPatches(
            split="test",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=None,
                min_involvement=self.config.min_invovlement,
                fold=self.config.fold
            ),
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
    
    def save_states(self):
        torch.save(
            {
                "model": self.model.state_dict(),
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

    def run_epoch(self, loader, train=True, desc="train"):
        with torch.no_grad() if not train else torch.enable_grad():
            self.model.train() if train else self.model.eval()
            
            
            for i, batch in enumerate(tqdm(loader, desc=desc)):
                images_aug_1, images_aug_2, images, labels, meta_data = batch
                images_aug_1 = images_aug_1.cuda()
                images_aug_2 = images_aug_2.cuda()
                images = images.cuda()
                labels = labels.cuda()
                
                # Zero gradients
                if train:
                    self.optimizer.zero_grad()
                
                (logits, 
                total_loss_avg,
                ce_loss_avg,
                byol_loss_avg
                ) = self.model(images_aug_1, images_aug_2, images, labels, training=train)

                # Optimizer step
                if train:
                    # Loss already backwarded in the model
                    self.optimizer.step()
                    self.scheduler.step()
                    wandb.log({"lr": self.scheduler.get_lr()[0]})
                             
                # Update metrics   
                self.metric_calculator.update(
                    batch_meta_data = meta_data,
                    logits = logits.detach().cpu().numpy(),
                    labels = labels.detach().cpu().numpy(),
                )
                
                # Log losses
                self.log_losses(total_loss_avg, ce_loss_avg, byol_loss_avg, desc)
            
            # Log metrics every epoch
            self.log_metrics(desc)
                
    
    def log_losses(self, total_loss_avg, ce_loss_avg, byol_loss_avg, desc):
        wandb.log({
            f"{desc}_loss": total_loss_avg,
            f"{desc}_ce_loss": ce_loss_avg,
            f"{desc}_byol_loss": byol_loss_avg,
            })
        
    def log_metrics(self, desc):
        metrics = self.metric_calculator.get_metrics()
        
        # Reset metrics for each epoch
        self.metric_calculator.reset()
        
        # Update best score
        (
            self.best_score_updated,
            self.best_score
            ) = self.metric_calculator.update_best_score(metrics, desc)
        
        # Log metrics
        wandb.log({
            f"{desc}/{key}": value for key, value in metrics.items()
            })
    
    def checkpoint(self):
        self.save_states()
        return super().checkpoint()


if __name__ == '__main__': 
    MT3Experiment.submit()