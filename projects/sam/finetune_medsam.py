import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass, asdict, field
from simple_parsing import ArgumentParser
from medAI.modeling import LayerNorm2d, Patchify
import medAI
from einops import rearrange, repeat
import wandb
from tqdm import tqdm
import submitit
from medAI.datasets import (
            CohortSelectionOptions,
            AlignedFilesDataset,
        )
import matplotlib.pyplot as plt
import medAI
import os
import logging 


DEFAULT_MEDSAM_CHECKPOINT = "/scratch/ssd004/scratch/pwilson/medsam_vit_b_cpu.pth"


@dataclass
class Config(medAI.utils.setup.BasicExperimentConfig):
    """Training configuration"""
    project: str = "finetune_medsam"

    epochs: int = 10
    lr: float = 0.00001
    batch_size: int = 1
    # whether to freeze the backbone and only finetune the mask decoder
    freeze_backbone: bool = True
    model_name: str = "medsam"
    wandb: bool = True
    debug: bool = False
    train_datasets: list[str] = field(default_factory=lambda: ["aligned_files", "nct"])
    test_datasets: list[str] = field(default_factory=lambda: ["aligned_files", "nct"])


def dice_loss(mask_probs, target_mask):
    intersection = (mask_probs * target_mask).sum()
    union = mask_probs.sum() + target_mask.sum()
    return 1 - 2 * intersection / union


def dice_score(mask_probs, target_mask):
    mask_probs = mask_probs > 0.5
    intersection = (mask_probs * target_mask).sum()
    union = mask_probs.sum() + target_mask.sum()
    return 2 * intersection / union


class Experiment(medAI.utils.setup.BasicExperiment):
    config_class = Config
    config: Config

    def __call__(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.epochs):
            self.save()
            logging.info(f"Epoch {self.epoch}")
            self.train_epoch(self.model, self.optimizer, self.train_loader)
            for name, test_loader in self.test_loaders.items():
                score = self.eval_epoch(self.model, test_loader, name=name)
                if score > self.best_score:
                    self.best_score = score
                    logging.info(f"New best score: {self.best_score}")
                    torch.save(self.model.medsam_model.state_dict(),
                        os.path.join(self.ckpt_dir, f"medsam-finetuned_image_encoder_{name}_{score}.pth")
                    )
            
    def setup(self):
        super().setup()
        self.setup_data()
        
        if 'experiment.ckpt' in os.listdir(self.ckpt_dir):
            logging.info('Loading from checkpoint')
            state = torch.load(os.path.join(self.ckpt_dir, 'experiment.ckpt'))
        else: 
            logging.info('No checkpoint found. Starting from scratch.')
            state = None

        logging.info('Setting up model, optimizer, scheduler')
        self.model = MedSAMForFinetuning(freeze_backbone=self.config.freeze_backbone)
        self.model = self.model.cuda()
        if state is not None:
            self.model.load_state_dict(state["model"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=5 * len(self.train_loader),
            max_epochs=self.config.epochs * len(self.train_loader),
        )
        if state is not None:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])

        logging.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        logging.info(
            f"""Trainable parameters: 
{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"""
        )

        self.epoch = 0 if state is None else state["epoch"]
        self.best_score = 0 if state is None else state["best_score"]

    def setup_data(self):
        from torchvision.transforms import v2 as T
        from torchvision.datapoints import Mask, Image
        from torchvision.transforms import InterpolationMode

        class AlignedFilesTransform:
            """Transforms for the aligned files dataset"""
            def __init__(self, augment=False): 
                self.augment = augment

            def __call__(self, item):
                bmode = item["image"]
                bmode = T.ToTensor()(bmode)
                bmode = T.Resize((1024, 1024), antialias=True)(bmode)
                bmode = (bmode - bmode.min() ) / (bmode.max() - bmode.min())
                bmode = bmode.repeat(3, 1, 1)
                bmode = Image(bmode)

                mask = item["mask"]
                mask = mask.astype("uint8")
                mask = T.ToTensor()(mask).float() 
                mask = T.Resize(
                    (1024, 1024), antialias=False, interpolation=InterpolationMode.NEAREST
                )(mask)
                mask = Mask(mask)

                if self.augment: 
                    augmentation = T.Compose([
                    T.RandomApply([T.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
                    T.RandomApply([T.RandomResizedCrop(1024, scale=(0.8, 1.0))], p=0.5),
                ])
                    bmode, mask = augmentation(bmode, Mask(mask))

                return bmode, mask 
            
        class NCTTransform:
            def __init__(self, augment=False): 
                self.augment = augment
            def __call__(self, item): 
                bmode = item["bmode"]
                bmode = np.flip(bmode, axis=0).copy()
                bmode = T.ToTensor()(bmode)
                bmode = T.Resize((1024, 1024), antialias=True)(bmode)
                bmode = (bmode - bmode.min() ) / (bmode.max() - bmode.min())
                bmode = bmode.repeat(3, 1, 1)
                bmode = Image(bmode)
                mask = item["prostate_mask"]
                mask = np.flip(mask, axis=0).copy()
                mask = mask.astype("uint8")
                mask = T.ToTensor()(mask).float() * 255 
                mask = T.Resize(
                    (1024, 1024), antialias=False, interpolation=InterpolationMode.NEAREST
                )(mask)
                mask = Mask(mask)
                if self.augment: 
                    augmentation = T.Compose([
                    T.RandomApply([T.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
                    T.RandomApply([T.RandomResizedCrop(1024, scale=(0.8, 1.0))], p=0.5),
                ])
                    bmode, mask = augmentation(bmode, Mask(mask))

                return bmode, mask

        logging.info('Setting up datasets')
        train_datasets = []
        if 'aligned_files' in self.config.train_datasets:
            train_datasets.append(
                AlignedFilesDataset(
                    split="train",
                    transform=AlignedFilesTransform(augment=True),
            ))
        if 'nct' in self.config.train_datasets:
            train_datasets.append(
                ExactNCT2013BModeImagesWithManualProstateSegmenation(
                    split="train",
                    transform=NCTTransform(augment=True),
                    cohort_selection_options=CohortSelectionOptions(),
            ))
        train_ds = torch.utils.data.ConcatDataset(train_datasets)
       
        test_datasets = {}
        if 'aligned_files' in self.config.test_datasets:
            test_datasets['aligned_files'] = AlignedFilesDataset(
                split="test",
                transform=AlignedFilesTransform(augment=False),
            )
        if 'nct' in self.config.test_datasets:
            test_datasets['nct'] = ExactNCT2013BModeImagesWithManualProstateSegmenation(
                split="test",
                transform=AlignedFilesTransform(augment=False),
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

    @torch.no_grad()
    def show_example(self, batch): 
        self.model.eval()
        X, y = batch 
        X = X.cuda()
        y = y.cuda()

        mask_logits = self.model(X)
        loss, _dice_score = self.model.get_loss_and_score(mask_logits, y)
        _dice_score = _dice_score.item()

        pred = mask_logits.sigmoid().detach().cpu().numpy()[0][0]
        mask = y[0, 0, :, :].detach().cpu().numpy()
        image =  X[0, 0, :, :].detach().cpu().numpy()

        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        [ax.set_axis_off() for ax in ax.flatten()]
        ax[0].imshow(image)
        ax[1].imshow(mask)
        ax[2].imshow(pred)
        ax[2].set_title(f'Dice score: {_dice_score:.3f}')
        ax[3].imshow(pred > 0.5)
        
    def train_epoch(self, model, optimizer, loader):
        model.train()

        dice = []
        acc_steps = 1
        for i, batch in enumerate(tqdm(loader)):
            X, y = batch
            X = X.cuda()
            y = y.cuda()
            
            mask_logits = self.model(X)
            loss, score = self.model.get_loss_and_score(mask_logits, y)
            score = score.item()

            loss.backward()

            if acc_steps % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()
                acc_steps = 1
            else:
                acc_steps += 1

            dice.append(score)

            self.scheduler.step()
            wandb.log({"train_loss": loss, "lr": self.scheduler.get_lr()[0]})

            if i % 100 == 0:
                self.show_example(batch)
                wandb.log(
                    {
                        "train_example": wandb.Image(
                            plt, caption=f"Epoch {self.epoch}, batch {i}"
                        )
                    }
                )
                plt.close()

            if self.config.debug and i > 10:
                break

        wandb.log(
            {
                "train_dice": sum(dice) / len(dice)
            }
        )

    @torch.no_grad()
    def eval_epoch(self, model, loader, name="test"):
        model.eval()

        dice = []
        for i, batch in enumerate(tqdm(loader)):
            X, y = batch
            X = X.cuda()
            y = y.cuda()
            
            mask_logits = self.model(X)
            score = self.model.get_loss_and_score(mask_logits, y)[1]

            dice.append(score.item())

            if i % 100 == 0:
                self.show_example(batch)
                wandb.log(
                    {
                        f"{name}_example": wandb.Image(
                            plt, caption=f"Epoch {self.epoch}, batch {i}"
                        )
                    }
                )
                plt.close()
            if self.config.debug and i > 10:
                break

        wandb.log(
            {
                f"{name}_dice": sum(dice) / len(dice)
            }
        )

        return sum(dice) / len(dice)
    
    def checkpoint(self):
        self.save()
        return super().checkpoint()

    
class MedSAMForFinetuning(nn.Module): 
    def __init__(self, medsam_checkpoint=DEFAULT_MEDSAM_CHECKPOINT, freeze_backbone=True): 
        super().__init__()
        from segment_anything import sam_model_registry
        self.medsam_model = sam_model_registry["vit_b"](checkpoint=DEFAULT_MEDSAM_CHECKPOINT)
        self.medsam_model.load_state_dict(
            torch.load(medsam_checkpoint, map_location="cpu")
        )
        if freeze_backbone:
            for param in self.medsam_model.image_encoder.parameters(): 
                param.requires_grad = False
            for param in self.medsam_model.prompt_encoder.parameters():
                param.requires_grad = False

    def forward(self, image): 
        image_feats = self.medsam_model.image_encoder(image)
        sparse_embedding, dense_embedding = self.medsam_model.prompt_encoder(
            None, None, None # no prompt - find prostate
        )
        mask_logits = self.medsam_model.mask_decoder.forward(
            image_feats, self.medsam_model.prompt_encoder.get_dense_pe(), 
            sparse_embedding, dense_embedding, multimask_output=False
        )[0]
        return mask_logits

    def get_loss_and_score(self, mask_logits, gt_mask): 
        B, C, H, W = mask_logits.shape

        gt_mask = torch.nn.functional.interpolate(gt_mask.float(), size=(H, W))

        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            mask_logits, gt_mask
        )
        _dice_loss = dice_loss(mask_logits.sigmoid(), gt_mask)
        loss = ce_loss + _dice_loss

        _dice_score = dice_score(mask_logits.sigmoid(), gt_mask)
        return loss, _dice_score
        

if __name__ == "__main__":
    Experiment.submit()
