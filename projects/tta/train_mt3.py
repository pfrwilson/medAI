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

from einops import rearrange, repeat
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging 


@dataclass
class Config(BasicExperimentConfig):
    """Configuration for the experiment."""
    project: str = "tta"
    epochs: int = 10 


class MyExperiment(BasicExperiment): 
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
        super().setup()
        if 'experiment.ckpt' in os.listdir(self.ckpt_dir):
            logging.info('Loading from checkpoint')
            state = torch.load(os.path.join(self.ckpt_dir, 'experiment.ckpt'))
        else: 
            logging.info('No checkpoint found. Starting from scratch.')
            state = None

        logging.info('Setting up datasets')
        train_ds = AlignedFilesDataset(
            split="train",
            transform=self.transform1,
        )
        train_ds = train_ds + ExactNCT2013BModeImagesWithManualProstateSegmenation(
            split="train",
            transform=self.transform2,
            cohort_selection_options=CohortSelectionOptions(),
        )
        train_ds = train_ds + ExactNCT2013BModeImagesWithManualProstateSegmenation(
            split="val",
            transform=self.transform2,
            cohort_selection_options=CohortSelectionOptions(),
        )
        
        test_datasets = {}
        test_datasets['aligned_files'] = AlignedFilesDataset(
            split="test",
            transform=lambda item: self.transform1(item, augment=False),
        )
        test_datasets['nct'] = ExactNCT2013BModeImagesWithManualProstateSegmenation(
            split="test",
            transform=lambda item: self.transform2(item),
            cohort_selection_options=CohortSelectionOptions(),
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        self.test_loaders = {
            name: DataLoader(
                test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
            )
            for name, test_ds in test_datasets.items()
        }

        logging.info('Setting up model, optimizer, scheduler')
        self.model = MedSAMForFinetuning(freeze_backbone=self.config.freeze_backbone)
        self.model = self.model.cuda()
        if state is not None:
            self.model.load_state_dict(state["model"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        if state is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=5 * len(self.train_loader),
            max_epochs=self.config.epochs * len(self.train_loader),
        )
        if state is not None:
            self.scheduler.load_state_dict(state["scheduler"])

        logging.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        logging.info(
            f"""Trainable parameters: 
{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"""
        )

        self.epoch = 0 if state is None else state["epoch"]
        self.best_score = 0 if state is None else state["best_score"]

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
                
                optimizer.zero_grad()
                
                (logits, 
                total_loss_avg,
                ce_loss_avg,
                byol_loss_avg
                ) = self.model(images_aug_1, images_aug_2, images, labels, training=True)

                # Loss already backwarded in the model
                optimizer.step()
                self.scheduler.step()
                
                wandb.log({
                    "train_loss": total_loss_avg,
                    "train_ce_loss": ce_loss_avg,
                    "train_byol_loss": byol_loss_avg,
                    "lr": self.scheduler.get_lr()[0]
                    })

    @torch.no_grad()
    def eval_epoch(self, model, loader, name="test"):
        model.eval()

        for i, batch in enumerate(tqdm(loader)):
            images_aug_1, images_aug_2, images, labels = batch
            images_aug_1 = images_aug_1.cuda()
            images_aug_2 = images_aug_2.cuda()
            images = images.cuda()
            labels = labels.cuda()
                    
            (logits, 
             total_loss_avg,
             ce_loss_avg,
             byol_loss_avg
             ) = self.model(images_aug_1, images_aug_2, images, labels, training=False)
            
            wandb.log({
                f"{name}_loss": total_loss_avg,
                f"{name}_ce_loss": ce_loss_avg,
                f"{name}_byol_loss": byol_loss_avg,
                })
    
    def checkpoint(self):
        self.save()
        return super().checkpoint()

    def setup(self): 
        super().setup()
        from torch.utils.data import DataLoader
        self.train_loader = DataLoader(range(100), batch_size=10, shuffle=True)

    def __call__(self): 
        self.setup()
        for epoch in range(self.config.epochs): 
            logging.info(f"Epoch {epoch}")
            for batch in self.train_loader: 
                wandb.log({"sum": sum(batch)})


if __name__ == '__main__': 
    MyExperiment.submit()