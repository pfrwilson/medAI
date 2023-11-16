import os
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
from medAI.datasets import (
            ExactNCT2013BModeImagesWithManualProstateSegmenation,
            CohortSelectionOptions,
            AlignedFilesDataset,
        )

from .models.model import MT3Model, MT3Config

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
    project: str = "MT3"
    epochs: int = 10 
    batch_size: int = 1
    fold: int = 0
    use_augmentation: bool = False
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
            
            #TODO: save model
            # torch.save(self.model.medsam_model.state_dict(), 
            #     os.path.join(self.ckpt_dir, f"medsam-finetuned_image_encoder_epoch={self.epoch}.pth")
            # )
            
    def setup(self):
        # logging setup
        super().setup()
        self.setup_data()

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
        from torchvision.datapoints import Image, Mask

        class Transform:
            def __init__(self, augment=False):
                self.augment = augment
                self.size = (256, 256)
            
            def __call__(self, item):
                bmode = item["bmode"]
                bmode = T.ToTensor()(bmode)
                bmode = T.Resize(self.size, antialias=True)(bmode)
                bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
                bmode = bmode.repeat(3, 1, 1)
                bmode = Image(bmode)

                needle_mask = item["needle_mask"]
                needle_mask = T.ToTensor()(needle_mask).float() * 255
                needle_mask = T.Resize(
                    self.size, antialias=False, interpolation=InterpolationMode.NEAREST
                )(needle_mask)
                needle_mask = Mask(needle_mask)

                prostate_mask = item["prostate_mask"]
                prostate_mask = T.ToTensor()(prostate_mask).float() * 255
                prostate_mask = T.Resize(
                    self.size, antialias=False, interpolation=InterpolationMode.NEAREST
                )(prostate_mask)
                prostate_mask = Mask(prostate_mask)

                if self.augment:
                    bmode, needle_mask, prostate_mask = T.RandomAffine(
                        degrees=0, translate=(0.1, 0.1)
                    )(bmode, needle_mask, prostate_mask)

                label = torch.tensor(item["grade"] != "Benign").long()
                return bmode, needle_mask, prostate_mask, label

        from medAI.datasets import ExactNCT2013BModeImages, CohortSelectionOptions, ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation

        train_ds = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split="train",
            transform=Transform(augment=self.config.use_augmentation),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=1,
                min_involvement=40,
                remove_benign_from_positive_patients=True,
                fold=self.config.fold,
            ),
        )
        test_ds = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split="test",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=None, min_involvement=40, fold=self.config.fold
            ),
        )
        if self.config.debug:
            train_ds = torch.utils.data.Subset(train_ds, torch.arange(0, 100))
            test_ds = torch.utils.data.Subset(test_ds, torch.arange(0, 100))

        self.train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )

    def save(self):
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

            core_probs = []
            core_labels = []
            patch_probs = []
            patch_labels = []
            
            for i, batch in enumerate(tqdm(loader, desc=desc)):
                images_aug_1, images_aug_2, images, labels = batch
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
                
                wandb.log({
                    f"{desc}_loss": total_loss_avg,
                    f"{desc}_ce_loss": ce_loss_avg,
                    f"{desc}_byol_loss": byol_loss_avg,
                    })

    def checkpoint(self):
        self.save()
        return super().checkpoint()



    def transform1(self, item, augment=True):
        pass
        # """Transforms for the aligned files dataset"""
        # from torchvision import transforms as T
        
        # from torchvision.transforms import InterpolationMode
        # bmode = item["image"]
        # bmode = T.ToTensor()(bmode)
        # bmode = T.Resize((1024, 1024), antialias=True)(bmode)
        # bmode = (bmode - bmode.min() ) / (bmode.max() - bmode.min())
        # bmode = bmode.repeat(3, 1, 1)
        # mask = item["mask"]
        # mask = mask.astype("uint8")
        # mask = T.ToTensor()(mask).float() 
        # mask = T.Resize(
        #     (1024, 1024), antialias=False, interpolation=InterpolationMode.NEAREST
        # )(mask)

        # if augment: 
        #     bmode, mask = self.augment(bmode, mask)

        # # label = torch.tensor(item["grade"] != "Benign").long()
        # return bmode, mask

    def augment(self, bmode, mask): 
        pass
        # from torchvision.datapoints import Mask
        # from torchvision.transforms.v2 import RandomResizedCrop, RandomApply, Compose, RandomAffine
        # import torchvision
        # torchvision.disable_beta_transforms_warning()

        # augmentation = Compose([
        #     RandomApply([RandomResizedCrop(1024, scale=(0.8, 1.0))], p=0.5),
        #     RandomApply([RandomAffine(0, translate=(0.2, 0.2))], p=0.3),
        # ])
        # bmode, mask = augmentation(bmode, Mask(mask))
        # return bmode, mask

    def transform2(self, item):
        pass
        # from torchvision import transforms as T
        # from torchvision.transforms import InterpolationMode
        # bmode = item["bmode"]
        # bmode = np.flip(bmode, axis=0).copy()
        # bmode = T.ToTensor()(bmode)
        # bmode = T.Resize((1024, 1024), antialias=True)(bmode)
        # bmode = (bmode - bmode.min() ) / (bmode.max() - bmode.min())
        # bmode = bmode.repeat(3, 1, 1)
        # mask = item["prostate_mask"]
        # mask = np.flip(mask, axis=0).copy()
        # mask = mask.astype("uint8")
        # mask = T.ToTensor()(mask).float() * 255 
        
        # mask = T.Resize(
        #     (1024, 1024), antialias=False, interpolation=InterpolationMode.NEAREST
        # )(mask)
        # return bmode, mask
      

if __name__ == '__main__': 
    MT3Experiment.submit()