import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import torch
import torch.nn as nn
import typing as tp
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
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


@dataclass
class FeatureExtractorConfig:
    model_name: str = 'resnet10t'
    num_classes: int = 2
    in_chans: int = 1
    features_only: bool = False # return features only, not logits
    num_groups: int = 8
    
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
class BaselineConfig(BasicExperimentConfig):
    """Configuration for the experiment."""
    # exp_dir: str = "./projects/tta/logs/first_experiment_test" 
    name: str = "baseline_group_norm_res10"
    group: str = None
    project: str = "tta" 
    entity: str = "mahdigilany"
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    
    epochs: int = 50
    batch_size: int = 32
    fold: int = 0
    
    min_invovlement: int = 40
    needle_mask_threshold: float = 0.5
    prostate_mask_threshold: float = 0.5
    patch_size_mm: tp.Tuple[float, float] = (5, 5)
    benign_to_cancer_ratio_test: tp.Optional[float] = 1.0
    
    model_config: FeatureExtractorConfig = FeatureExtractorConfig()
    optimizer_config: OptimizerConfig = OptimizerConfig()
    
        
class BaselineExperiment(BasicExperiment): 
    config_class = BaselineConfig
    config: BaselineConfig

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

        if "experiment.ckpt" in os.listdir(self.ckpt_dir) and self.config.resume:
            state = torch.load(os.path.join(self.ckpt_dir, "experiment.ckpt"))
            logging.info(f"Resuming from epoch {state['epoch']}")
        else:
            state = None

        logging.info('Setting up model, optimizer, scheduler')
        self.model = self.setup_model()
        self.model = self.model.cuda()
        
        self.optimizer = create_optimizer(self.config.optimizer_config, self.model)
        # import torch.optim as optim
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
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
        # from torchvision.datapoints import Image as TVImage
        from torchvision.tv_tensors import Image as TVImage

        class Transform:
            def __init__(selfT, augment=False):
                selfT.augment = augment
                selfT.size = (256, 256)
            
            def __call__(selfT, item):
                patch = item.pop("patch")
                patch = (patch - patch.min()) / (patch.max() - patch.min())
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


        from datasets.datasets import ExactNCT2013RFImagePatches, CohortSelectionOptions, PatchOptions
        
        train_ds = ExactNCT2013RFImagePatches(
            split="train",
            transform=Transform(augment=False),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=1,
                min_involvement=self.config.min_invovlement,
                remove_benign_from_positive_patients=True,
                fold=self.config.fold,
            ),
            patch_options=PatchOptions(
                patch_size_mm=self.config.patch_size_mm,
                needle_mask_threshold=self.config.needle_mask_threshold,
                prostate_mask_threshold=self.config.prostate_mask_threshold,
            ),
            debug=self.config.debug,
        )
        
        val_ds = ExactNCT2013RFImagePatches(
            split="val",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=self.config.benign_to_cancer_ratio_test,
                min_involvement=self.config.min_invovlement,
                fold=self.config.fold
            ),
            patch_options=PatchOptions(
                patch_size_mm=self.config.patch_size_mm,
                needle_mask_threshold=self.config.needle_mask_threshold,
                prostate_mask_threshold=self.config.prostate_mask_threshold,
            ),
            debug=self.config.debug,
        )
                
        test_ds = ExactNCT2013RFImagePatches(
            split="test",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=self.config.benign_to_cancer_ratio_test,
                min_involvement=self.config.min_invovlement,
                fold=self.config.fold
            ),
            patch_options=PatchOptions(
                patch_size_mm=self.config.patch_size_mm,
                needle_mask_threshold=self.config.needle_mask_threshold,
                prostate_mask_threshold=self.config.prostate_mask_threshold,
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

        self.test_loaders = {
            "val": self.val_loader,
            "test": self.test_loader
        }
        
    def setup_metrics(self):
        self.metric_calculator = MetricCalculator()
    
    def setup_model(self):
        # Create the model
        model: nn.Module = timm.create_model(
            self.config.model_config.model_name,
            num_classes=self.config.model_config.num_classes,
            in_chans=self.config.model_config.in_chans,
            features_only=self.config.model_config.features_only,
            norm_layer=lambda channels: nn.GroupNorm(
                num_groups=self.config.model_config.num_groups, num_channels=channels
                )
            )
        return model
    
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
                    logits = logits.detach().cpu(),
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
        
        # Reset metrics for each epoch
        self.metric_calculator.reset()
        
        # Update best score
        (
            self.best_score_updated,
            self.best_score
            ) = self.metric_calculator.update_best_score(metrics, desc)
        
        # Log metrics
        metrics_dict = {
            f"{desc}/{key}": value for key, value in metrics.items()
            }
        metrics_dict.update({"epoch": self.epoch})
        wandb.log(
            metrics_dict,
            commit=True
            )
    
    def checkpoint(self):
        self.save_states()
        return super().checkpoint()



if __name__ == '__main__': 
    BaselineExperiment.submit()