import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass, asdict
from simple_parsing import ArgumentParser, Serializable, choice
from simple_parsing.helpers import Serializable
from medAI.modeling import LayerNorm2d, Patchify
import medAI
from medAI.utils.setup import BasicExperiment, BasicExperimentConfig
from segment_anything import sam_model_registry
from einops import rearrange, repeat
import wandb
from tqdm import tqdm
import submitit
import matplotlib.pyplot as plt
import os
import logging
import numpy as np
import typing as tp
from abc import ABC, abstractmethod
from medsam_cancer_detection_v2_model_registry import (
    model_registry,
    MaskedPredictionModule,
)
from typing import Any
import typing as tp
import warnings
warnings.filterwarnings("ignore")


@dataclass
class Config(BasicExperimentConfig, Serializable):
    """Training configuration"""

    project: str = "medsam_cancer_detection_v3"
    fold: int = 0
    n_folds: int = 5
    benign_cancer_ratio_for_training: float | None = None
    epochs: int = 30
    optimizer: tp.Literal["adam", "sgdw"] = "adam"
    use_augmentation: bool = False
    loss: tp.Literal["basic_ce", "involvement_tolerant_loss"] = "basic_ce"
    lr: float = 0.00001
    wd: float = 0.0
    batch_size: int = 1
    model_config: Any = model_registry.get_simple_parsing_subgroups()
    debug: bool = False
    accumulate_grad_steps: int = 8
    min_involvement_pct_training: float = 40.0
    prostate_threshold: float = 0.5
    needle_threshold: float = 0.5       

    # probability of sampling a batch from the segmentation dataset (vs the detection dataset)
    # during training
    segmentation_batch_prob: float = 0.5


class Experiment(BasicExperiment):
    config_class = Config
    config: Config

    def setup(self):
        # logging setup
        super().setup()
        (
            self.train_loader_detection,
            self.val_loader_detection,
            self.test_loader_detection,
        ) = self._setup_data_t1()
        (
            self.train_loader_segmentation,
            self.test_loader_segmentation,
        ) = self._setup_data_t2()

        if "experiment.ckpt" in os.listdir(self.ckpt_dir):
            state = torch.load(os.path.join(self.ckpt_dir, "experiment.ckpt"))
        else:
            state = None

        # Setup model
        image_encoder, prompt_encoder_1, mask_decoder_1 = build_medsam_model_components()
        _, prompt_encoder_2, mask_decoder_2 = build_medsam_model_components()

        self.detection_model = MedSAMSegmentator(
            image_encoder, prompt_encoder_1, mask_decoder_1
        ).cuda()
        self.segmentation_model = MedSAMSegmentator(
            image_encoder, prompt_encoder_2, mask_decoder_2
        ).cuda()

        if state is not None:
            self.detection_model.load_state_dict(state["detection_model"])
            self.segmentation_model.load_state_dict(state["segmentation_model"])

        def _make_optimizer_and_scheduler(model, n_steps_per_epoch):
            parameters = [{
                "params": model.image_encoder.parameters(),
                "lr": self.config.lr / 10,
            }, {
                "params": model.prompt_encoder.parameters(),
                "lr": self.config.lr,
            }, {
                "params": model.mask_decoder.parameters(),
                "lr": self.config.lr,
            }]
            match self.config.optimizer:
                # make lr less for encoder
                case "adam":
                    opt = optim.Adam(
                        parameters,
                        lr=self.config.lr,
                        weight_decay=self.config.wd,
                    )
                case "sgdw":
                    opt = optim.SGD(
                        parameters,
                        lr=self.config.lr,
                        momentum=0.9,
                        weight_decay=self.config.wd,
                    )
            scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
                opt,
                warmup_epochs=5 * n_steps_per_epoch,
                max_epochs=self.config.epochs * n_steps_per_epoch,
            )
            return opt, scheduler

        self.optimizer_detect, self.scheduler_detect = _make_optimizer_and_scheduler(
            self.detection_model, len(self.train_loader_detection)
        )
        self.optimizer_segment, self.scheduler_segment = _make_optimizer_and_scheduler(
            self.detection_model, int(len(self.train_loader_segmentation) * self.config.segmentation_batch_prob)
        )

        if state is not None:
            self.optimizer_detect.load_state_dict(state["optimizer_detect"])
            self.scheduler_detect.load_state_dict(state["scheduler_detect"])
            self.optimizer_segment.load_state_dict(state["optimizer_segment"])
            self.scheduler_segment.load_state_dict(state["scheduler_segment"])

        self.masked_prediction_module_train = MaskedPredictionModule(
            needle_mask_threshold=self.config.needle_threshold,
            prostate_mask_threshold=self.config.prostate_threshold,
        )
        self.masked_prediction_module_test = MaskedPredictionModule()

        self.epoch = 0 if state is None else state["epoch"]
        self.best_score = 0 if state is None else state["best_score"]

    def _setup_data_t1(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import v2 as T
        from torchvision.tv_tensors import Image, Mask

        class Transform:
            def __init__(self, augment=False):
                self.augment = augment

            def __call__(self, item):
                bmode = item["bmode"]
                bmode = np.flip(bmode, axis=0).copy()
                bmode = T.ToTensor()(bmode)
                bmode = T.Resize((1024, 1024), antialias=True)(bmode)
                bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
                bmode = bmode.repeat(3, 1, 1)
                bmode = Image(bmode)

                needle_mask = item["needle_mask"]
                needle_mask = np.flip(needle_mask, axis=0).copy()
                needle_mask = T.ToTensor()(needle_mask).float() * 255
                needle_mask = T.Resize(
                    (1024, 1024),
                    antialias=False,
                    interpolation=InterpolationMode.NEAREST,
                )(needle_mask)
                needle_mask = Mask(needle_mask)

                prostate_mask = item["prostate_mask"]
                prostate_mask = np.flip(prostate_mask, axis=0).copy()
                prostate_mask = T.ToTensor()(prostate_mask).float() * 255
                prostate_mask = T.Resize(
                    (1024, 1024),
                    antialias=False,
                    interpolation=InterpolationMode.NEAREST,
                )(prostate_mask)
                prostate_mask = Mask(prostate_mask)

                if self.augment:
                    bmode, needle_mask, prostate_mask = T.RandomAffine(
                        degrees=0, translate=(0.2, 0.2)
                    )(bmode, needle_mask, prostate_mask)

                label = torch.tensor(item["grade"] != "Benign").long()
                pct_cancer = item["pct_cancer"]
                if np.isnan(pct_cancer):
                    pct_cancer = 0
                involvement = torch.tensor(pct_cancer / 100).float()
                return bmode, needle_mask, prostate_mask, label, involvement

        from medAI.datasets import (
            ExactNCT2013BModeImages,
            CohortSelectionOptions,
            ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation,
        )

        train_ds = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split="train",
            transform=Transform(augment=self.config.use_augmentation),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=self.config.benign_cancer_ratio_for_training,
                min_involvement=self.config.min_involvement_pct_training,
                remove_benign_from_positive_patients=True,
                fold=self.config.fold,
                n_folds=self.config.n_folds,
            ),
        )
        val_ds = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split="val",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=None,
                min_involvement=None,
                fold=self.config.fold,
                n_folds=self.config.n_folds,
            ),
        )
        test_ds = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split="test",
            transform=Transform(),
            cohort_selection_options=CohortSelectionOptions(
                benign_to_cancer_ratio=None,
                min_involvement=None,
                fold=self.config.fold,
                n_folds=self.config.n_folds,
            ),
        )
        if self.config.debug:
            train_ds = torch.utils.data.Subset(train_ds, list(range(0, 10)))
            val_ds = torch.utils.data.Subset(val_ds, list(range(0, 10)))
            test_ds = torch.utils.data.Subset(test_ds, list(range(0, 10)))

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )
        return train_loader, val_loader, test_loader

    def _setup_data_t2(self):
        from torchvision.transforms import v2 as T
        from torchvision.tv_tensors import Image, Mask
        from torchvision.transforms import InterpolationMode
        from medAI.datasets import ExactNCT2013BmodeImagesWithManualProstateSegmentation, CohortSelectionOptions

        class AlignedFilesTransform:
            """Transforms for the aligned files dataset"""

            def __init__(self, augment=False):
                self.augment = augment

            def __call__(self, item):
                bmode = item["image"]
                bmode = T.ToTensor()(bmode)
                bmode = T.Resize((1024, 1024), antialias=True)(bmode)
                bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
                bmode = bmode.repeat(3, 1, 1)
                bmode = Image(bmode)

                mask = item["mask"]
                mask = mask.astype("uint8")
                mask = T.ToTensor()(mask).float()
                mask = T.Resize(
                    (1024, 1024),
                    antialias=False,
                    interpolation=InterpolationMode.NEAREST,
                )(mask)
                mask = Mask(mask)

                if self.augment:
                    augmentation = T.Compose(
                        [
                            T.RandomApply(
                                [T.RandomAffine(0, translate=(0.2, 0.2))], p=0.5
                            ),
                            T.RandomApply(
                                [T.RandomResizedCrop(1024, scale=(0.8, 1.0))], p=0.5
                            ),
                        ]
                    )
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

        from medAI.datasets import AlignedFilesDataset

        train_ds = AlignedFilesDataset(
            split="train",
            transform=AlignedFilesTransform(augment=True),
        )
        train_ds = train_ds + ExactNCT2013BmodeImagesWithManualProstateSegmentation(
            split="train",
            transform=NCTTransform(augment=True),
            cohort_selection_options=CohortSelectionOptions(
                fold=self.config.fold,
                n_folds=self.config.n_folds,
            ),
        )
        test_ds = AlignedFilesDataset(
            split="test",
            transform=AlignedFilesTransform(),
        )
        if self.config.debug: 
            train_ds = torch.utils.data.Subset(train_ds, torch.arange(0, 10))
            test_ds = torch.utils.data.Subset(test_ds, torch.arange(0, 10))

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )
        return train_loader, test_loader

    def __call__(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.epochs):
            logging.info(f"Epoch {self.epoch}")
            self.training = True
            self.train_epoch()
            self.training = False
            val_metrics = self.validation_epoch()
            tracked_metric = val_metrics["val_detection/core_auc_high_involvement"]
            if tracked_metric > self.best_score:
                self.best_score = tracked_metric
                logging.info(f"New best score: {self.best_score}")
                self.test_epoch()

    def train_epoch(self): 
        detection_loader_iter = iter(self.train_loader_detection)
        segmentation_loader_iter = iter(self.train_loader_segmentation)

        detection_logic = PCaDetectionTrainingLogic(
            self.config,
            self.detection_model,
            self.optimizer_detect,
            self.scheduler_detect,
            prefix="train_detection",
            epoch=self.epoch,
            log_image_interval=100 if self.training else 10,
        )

        segmentation_logic = SegmentationTrainingLogic(
            self.config,
            self.segmentation_model,
            self.optimizer_segment,
            self.scheduler_segment,
            prefix="train_segmentation",
            epoch=self.epoch,
            log_image_interval=100 if self.training else 10,
        )

        pbar = tqdm(total=len(self.train_loader_detection), desc="train_detection")
        while True: 
            if np.random.rand() < self.config.segmentation_batch_prob: 
                try: 
                    batch = next(segmentation_loader_iter)
                except StopIteration: 
                    segmentation_loader_iter = iter(self.train_loader_segmentation)
                    batch = next(segmentation_loader_iter)
                segmentation_logic.step(batch, train=True)
            else: 
                try: 
                    batch = next(detection_loader_iter)
                    detection_logic.step(batch, train=True)
                    pbar.update(1)
                except StopIteration: 
                    break
            
        detection_metrics = detection_logic.finish_epoch()
        segmentation_metrics = segmentation_logic.finish_epoch()
        metrics = {**detection_metrics, **segmentation_metrics}

        return metrics

    def validation_epoch(self): 
        detection_logic = PCaDetectionTrainingLogic(
            self.config,
            self.detection_model,
            self.optimizer_detect,
            self.scheduler_detect,
            prefix="val_detection",
            epoch=self.epoch,
            log_image_interval=100 if self.training else 10,
        )

        for batch in tqdm(self.test_loader_detection, desc="val_detection"):
            detection_logic.step(batch, train=False)

        detection_metrics = detection_logic.finish_epoch()
        return detection_metrics
    
    def test_epoch(self):
        detection_logic = PCaDetectionTrainingLogic(
            self.config,
            self.detection_model,
            self.optimizer_detect,
            self.scheduler_detect,
            prefix="test_detection",
            epoch=self.epoch,
            log_image_interval=100 if self.training else 10,
        )

        for batch in tqdm(self.test_loader_detection, desc="test_detection"):
            detection_logic.step(batch, train=False)

        segmentation_logic = SegmentationTrainingLogic(
            self.config,
            self.segmentation_model,
            self.optimizer_segment,
            self.scheduler_segment,
            prefix="test_segmentation",
            epoch=self.epoch,
            log_image_interval=100 if self.training else 10,
        )

        for batch in tqdm(self.test_loader_segmentation, desc="test_segmentation"):
            segmentation_logic.step(batch, train=False)

        detection_metrics = detection_logic.finish_epoch()
        segmentation_metrics = segmentation_logic.finish_epoch()
        metrics = {**detection_metrics, **segmentation_metrics}

        return metrics

    @torch.no_grad()
    def show_example(self, batch):
        image, needle_mask, prostate_mask, label, involvement = batch
        image = image.cuda()
        needle_mask = needle_mask.cuda()
        prostate_mask = prostate_mask.cuda()
        label = label.cuda()

        logits = self.model(image)
        masked_prediction_module = (
            self.masked_prediction_module_train
            if self.training
            else self.masked_prediction_module_test
        )
        outputs = masked_prediction_module(logits, needle_mask, prostate_mask, label)
        # outputs: MaskedPredictionModule.Output = self.masked_prediction_module(logits, needle_mask, prostate_mask, label)
        core_prediction = outputs.core_predictions[0].item()

        pred = logits.sigmoid()

        needle_mask = needle_mask.cpu()
        prostate_mask = prostate_mask.cpu()
        logits = logits.cpu()
        pred = pred.cpu()
        image = image.cpu()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        [ax.set_axis_off() for ax in ax.flatten()]
        kwargs = dict(vmin=0, vmax=1, extent=(0, 46, 0, 28))

        ax[0].imshow(image[0].permute(1, 2, 0), **kwargs)
        prostate_mask = prostate_mask.cpu()
        ax[0].imshow(
            prostate_mask[0, 0], alpha=prostate_mask[0][0] * 0.3, cmap="Blues", **kwargs
        )
        ax[0].imshow(needle_mask[0, 0], alpha=needle_mask[0][0], cmap="Reds", **kwargs)
        ax[0].set_title(f"Ground truth label: {label[0].item()}")

        ax[1].imshow(pred[0, 0], **kwargs)

        if self.training:
            valid_loss_region = (
                needle_mask[0][0] > self.config.needle_threshold
            ).float() * (prostate_mask[0][0] > self.config.prostate_threshold).float()
        else:
            valid_loss_region = (prostate_mask[0][0] > 0.5).float() * (
                needle_mask[0][0] > 0.5
            ).float()

        alpha = torch.nn.functional.interpolate(
            valid_loss_region[None, None], size=(256, 256), mode="nearest"
        )[0, 0]
        ax[2].imshow(pred[0, 0], alpha=alpha, **kwargs)
        ax[2].set_title(f"Core prediction: {core_prediction:.3f}")

    def save(self):
        torch.save(
            {
                "detection_model": self.detection_model.state_dict(),
                "segmentation_model": self.segmentation_model.state_dict(),
                "optimizer_detect": self.optimizer_detect.state_dict(),
                "scheduler_detect": self.scheduler_detect.state_dict(),
                "optimizer_segment": self.optimizer_segment.state_dict(),
                "scheduler_segment": self.scheduler_segment.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
            },
            os.path.join(self.ckpt_dir, "experiment.ckpt"),
        )

    def checkpoint(self):
        self.save()
        return super().checkpoint()


class PCaDetectionTrainingLogic:
    def __init__(
        self,
        config: Config,
        model,
        optimizer,
        scheduler,
        prefix="train",
        epoch=None,
        log_image_interval=100,
    ):
        self.config = config
        self.acc_steps = 1
        self.prefix = prefix
        self.epoch = epoch
        self.log_image_interval = log_image_interval

        self.masked_prediction_module_train = MaskedPredictionModule(
            needle_mask_threshold=self.config.needle_threshold,
            prostate_mask_threshold=self.config.prostate_threshold,
        )
        self.masked_prediction_module_test = MaskedPredictionModule()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.total_steps = 0

        self.reset_epoch()

    def _train_step(self, batch):
        self.model.train()

        bmode, needle_mask, prostate_mask, label, involvement = batch

        bmode = bmode.cuda()
        needle_mask = needle_mask.cuda()
        prostate_mask = prostate_mask.cuda()
        label = label.cuda()

        heatmap_logits = self.model(bmode)
        BATCH_SIZE = heatmap_logits.shape[0]

        outputs: MaskedPredictionModule.Output = self.masked_prediction_module_train(
            heatmap_logits, needle_mask, prostate_mask, label
        )

        loss = self.compute_loss(heatmap_logits, outputs, involvement)

        loss.backward()
        if self.acc_steps % self.config.accumulate_grad_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.acc_steps = 1
        else:
            self.acc_steps += 1
        self.scheduler.step()
        wandb.log(
            {
                f"{self.prefix}_loss": loss,
                f"{self.prefix}_lr": self.scheduler.get_last_lr()[0],
            }
        )

        self.core_probs.append(outputs.core_predictions.detach().cpu())
        self.core_labels.append(outputs.core_labels.detach().cpu())
        self.patch_probs.append(outputs.patch_predictions.detach().cpu())
        self.patch_labels.append(outputs.patch_labels.detach().cpu())
        patch_predictions = (outputs.patch_predictions > 0.5).float()
        pred_involvement_batch = []
        for core_idx in outputs.core_indices.unique():
            pred_involvement_batch.append(
                patch_predictions[outputs.core_indices == core_idx].mean()
            )
        self.pred_involvement.append(torch.stack(pred_involvement_batch).detach().cpu())
        patch_involvement_i = torch.zeros_like(patch_predictions)
        for core_idx in range(len(involvement)):
            patch_involvement_i[outputs.core_indices == core_idx] = involvement[
                core_idx
            ]
        self.patch_involvement.append(patch_involvement_i.detach().cpu())
        self.gt_involvement.append(involvement)

    @torch.no_grad()
    def _val_step(self, batch):
        self.model.eval()

        bmode, needle_mask, prostate_mask, label, involvement = batch

        bmode = bmode.cuda()
        needle_mask = needle_mask.cuda()
        prostate_mask = prostate_mask.cuda()
        label = label.cuda()

        heatmap_logits = self.model(bmode)

        outputs: MaskedPredictionModule.Output = self.masked_prediction_module_train(
            heatmap_logits, needle_mask, prostate_mask, label
        )

        self.core_probs.append(outputs.core_predictions.detach().cpu())
        self.core_labels.append(outputs.core_labels.detach().cpu())
        self.patch_probs.append(outputs.patch_predictions.detach().cpu())
        self.patch_labels.append(outputs.patch_labels.detach().cpu())
        patch_predictions = (outputs.patch_predictions > 0.5).float()
        pred_involvement_batch = []
        for core_idx in outputs.core_indices.unique():
            pred_involvement_batch.append(
                patch_predictions[outputs.core_indices == core_idx].mean()
            )
        self.pred_involvement.append(torch.stack(pred_involvement_batch).detach().cpu())
        patch_involvement_i = torch.zeros_like(patch_predictions)
        for core_idx in range(len(involvement)):
            patch_involvement_i[outputs.core_indices == core_idx] = involvement[
                core_idx
            ]
        self.patch_involvement.append(patch_involvement_i.detach().cpu())
        self.gt_involvement.append(involvement)

    def step(self, batch, train=True):
        if train:
            self._train_step(batch)
        else:
            self._val_step(batch)

        self.total_steps += 1
        if self.total_steps % self.log_image_interval == 0:
            self.show_example(batch, train=train)
            wandb.log({f"{self.prefix}_example": wandb.Image(plt)})
            plt.close()

    def compute_loss(self, heatmap_logits, outputs, involvement):
        BATCH_SIZE = heatmap_logits.shape[0]

        match self.config.loss:
            case "basic_ce":
                loss = nn.functional.binary_cross_entropy(
                    outputs.patch_predictions, outputs.patch_labels
                )
            case "involvement_tolerant_loss":
                loss = torch.tensor(
                    0, dtype=torch.float32, device=heatmap_logits.device
                )
                for i in range(BATCH_SIZE):
                    patch_predictions_for_core = outputs.patch_predictions[
                        outputs.core_indices == i
                    ]
                    patch_labels_for_core = outputs.patch_labels[
                        outputs.core_indices == i
                    ]
                    involvement_for_core = involvement[i]
                    if patch_labels_for_core[0].item() == 0:
                        # core is benign, so label noise is assumed to be low
                        loss += nn.functional.binary_cross_entropy(
                            patch_predictions_for_core, patch_labels_for_core
                        )
                    elif involvement_for_core.item() > 0.65:
                        # core is high involvement, so label noise is assumed to be low
                        loss += nn.functional.binary_cross_entropy(
                            patch_predictions_for_core, patch_labels_for_core
                        )
                    else:
                        # core is of intermediate involvement, so label noise is assumed to be high.
                        # we should be tolerant of the model's "false positives" in this case.
                        pred_index_sorted_by_cancer_score = torch.argsort(
                            patch_predictions_for_core[:, 0], descending=True
                        )
                        patch_predictions_for_core = patch_predictions_for_core[
                            pred_index_sorted_by_cancer_score
                        ]
                        patch_labels_for_core = patch_labels_for_core[
                            pred_index_sorted_by_cancer_score
                        ]
                        n_predictions = patch_predictions_for_core.shape[0]
                        patch_predictions_for_core_for_loss = (
                            patch_predictions_for_core[
                                : int(n_predictions * involvement_for_core.item())
                            ]
                        )
                        patch_labels_for_core_for_loss = patch_labels_for_core[
                            : int(n_predictions * involvement_for_core.item())
                        ]
                        loss += nn.functional.binary_cross_entropy(
                            patch_predictions_for_core_for_loss,
                            patch_labels_for_core_for_loss,
                        )

            case _:
                raise ValueError(f"Unknown loss: {self.config.loss}")

        return loss

    def finish_epoch(self):
        desc = self.prefix
        epoch = self.epoch
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score, r2_score

        metrics = {}

        # core predictions
        core_probs = torch.cat(self.core_probs)
        core_labels = torch.cat(self.core_labels)
        metrics["core_auc"] = roc_auc_score(core_labels, core_probs)
        plt.hist(core_probs[core_labels == 0], bins=100, alpha=0.5, density=True)
        plt.hist(core_probs[core_labels == 1], bins=100, alpha=0.5, density=True)
        plt.legend(["Benign", "Cancer"])
        plt.xlabel(f"Probability of cancer")
        plt.ylabel("Density")
        plt.title(f"Core AUC: {metrics['core_auc']:.3f}")
        wandb.log(
            {
                f"{desc}_corewise_histogram": wandb.Image(
                    plt, caption="Histogram of core predictions"
                )
            }
        )
        plt.close()

        # involvement predictions
        pred_involvement = torch.cat(self.pred_involvement)
        gt_involvement = torch.cat(self.gt_involvement)
        metrics["involvement_r2"] = r2_score(gt_involvement, pred_involvement)
        plt.scatter(gt_involvement, pred_involvement)
        plt.xlabel("Ground truth involvement")
        plt.ylabel("Predicted involvement")
        plt.title(f"Involvement R2: {metrics['involvement_r2']:.3f}")
        wandb.log(
            {
                f"{desc}_involvement": wandb.Image(
                    plt, caption="Ground truth vs predicted involvement"
                )
            }
        )
        plt.close()

        # high involvement core predictions
        high_involvement = gt_involvement > 0.4
        benign = core_labels[:, 0] == 0
        keep = torch.logical_or(high_involvement, benign)
        if keep.sum() > 0:
            core_probs = core_probs[keep]
            core_labels = core_labels[keep]
            metrics["core_auc_high_involvement"] = roc_auc_score(
                core_labels, core_probs
            )
            plt.hist(core_probs[core_labels == 0], bins=100, alpha=0.5, density=True)
            plt.hist(core_probs[core_labels == 1], bins=100, alpha=0.5, density=True)
            plt.legend(["Benign", "Cancer"])
            plt.xlabel(f"Probability of cancer")
            plt.ylabel("Density")
            plt.title(
                f"Core AUC (high involvement): {metrics['core_auc_high_involvement']:.3f}"
            )
            wandb.log(
                {
                    f"{desc}/corewise_histogram_high_involvement": wandb.Image(
                        plt, caption="Histogram of core predictions"
                    )
                }
            )
            plt.close()

        self.reset_epoch()
        metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)
        return metrics

    def reset_epoch(self):
        self.core_probs = []
        self.core_labels = []
        self.patch_probs = []
        self.patch_labels = []
        self.pred_involvement = []
        self.gt_involvement = []
        self.patch_involvement = []

    @torch.no_grad()
    def show_example(self, batch, train=True):
        image, needle_mask, prostate_mask, label, involvement = batch
        image = image.cuda()
        needle_mask = needle_mask.cuda()
        prostate_mask = prostate_mask.cuda()
        label = label.cuda()

        logits = self.model(image)
        masked_prediction_module = (
            self.masked_prediction_module_train
            if train
            else self.masked_prediction_module_test
        )
        outputs = masked_prediction_module(logits, needle_mask, prostate_mask, label)
        # outputs: MaskedPredictionModule.Output = self.masked_prediction_module(logits, needle_mask, prostate_mask, label)
        core_prediction = outputs.core_predictions[0].item()

        pred = logits.sigmoid()

        needle_mask = needle_mask.cpu()
        prostate_mask = prostate_mask.cpu()
        logits = logits.cpu()
        pred = pred.cpu()
        image = image.cpu()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        [ax.set_axis_off() for ax in ax.flatten()]
        kwargs = dict(vmin=0, vmax=1, extent=(0, 46, 0, 28))

        ax[0].imshow(image[0].permute(1, 2, 0), **kwargs)
        prostate_mask = prostate_mask.cpu()
        ax[0].imshow(
            prostate_mask[0, 0], alpha=prostate_mask[0][0] * 0.3, cmap="Blues", **kwargs
        )
        ax[0].imshow(needle_mask[0, 0], alpha=needle_mask[0][0], cmap="Reds", **kwargs)
        ax[0].set_title(f"Ground truth label: {label[0].item()}")

        ax[1].imshow(pred[0, 0], **kwargs)

        if train:
            valid_loss_region = (
                needle_mask[0][0] > self.config.needle_threshold
            ).float() * (prostate_mask[0][0] > self.config.prostate_threshold).float()
        else:
            valid_loss_region = (prostate_mask[0][0] > 0.5).float() * (
                needle_mask[0][0] > 0.5
            ).float()

        alpha = torch.nn.functional.interpolate(
            valid_loss_region[None, None], size=(256, 256), mode="nearest"
        )[0, 0]
        ax[2].imshow(pred[0, 0], alpha=alpha, **kwargs)
        ax[2].set_title(f"Core prediction: {core_prediction:.3f}")


def dice_loss(mask_probs, target_mask):
    intersection = (mask_probs * target_mask).sum()
    union = mask_probs.sum() + target_mask.sum()
    return 1 - 2 * intersection / union


def dice_score(mask_probs, target_mask):
    mask_probs = mask_probs > 0.5
    intersection = (mask_probs * target_mask).sum()
    union = mask_probs.sum() + target_mask.sum()
    return 2 * intersection / union


class SegmentationTrainingLogic:
    def __init__(
        self,
        config: Config,
        model,
        optimizer,
        scheduler,
        prefix="train",
        epoch=None,
        log_image_interval=100,
    ):
        self.config = config
        self.acc_steps = 1
        self.prefix = prefix
        self.epoch = epoch
        self.log_image_interval = log_image_interval

        self.masked_prediction_module_train = MaskedPredictionModule(
            needle_mask_threshold=self.config.needle_threshold,
            prostate_mask_threshold=self.config.prostate_threshold,
        )
        self.masked_prediction_module_test = MaskedPredictionModule()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.total_steps = 0

        self.reset_epoch()

    def _train_step(self, batch):
        self.model.train()

        X, y = batch
        X = X.cuda()
        y = y.cuda()

        mask_logits = self.model(X)
        loss, score = self.get_loss_and_score(mask_logits, y)

        score = score.item()

        loss.backward()

        if self.acc_steps % 1 == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.acc_steps = 1
        else:
            self.acc_steps += 1

        self.dice.append(score)

        self.scheduler.step()
        wandb.log(
            {
                f"{self.prefix}_loss": loss,
                f"{self.prefix}_lr": self.scheduler.get_lr()[0],
            }
        )

    @torch.no_grad()
    def _val_step(self, batch):
        self.model.eval()

        X, y = batch
        X = X.cuda()
        y = y.cuda()

        mask_logits = self.model(X)
        loss, score = self.get_loss_and_score(mask_logits, y)

        score = score.item()

        self.dice.append(score)

        self.scheduler.step()

    def step(self, batch, train=True):
        if train:
            self._train_step(batch)
        else:
            self._val_step(batch)

        if self.total_steps % self.log_image_interval == 0:
            self.show_example(batch)
            wandb.log({f"{self.prefix}_example": wandb.Image(plt)})
            plt.close()

        self.total_steps += 1

    def finish_epoch(self):
        wandb.log({f"{self.prefix}_dice": sum(self.dice) / len(self.dice)})
        self.reset_epoch()
        return {}

    def reset_epoch(self):
        self.dice = []

    @torch.no_grad()
    def show_example(self, batch):
        self.model.eval()
        X, y = batch
        X = X.cuda()
        y = y.cuda()

        mask_logits = self.model(X)
        loss, _dice_score = self.get_loss_and_score(mask_logits, y)
        _dice_score = _dice_score.item()

        pred = mask_logits.sigmoid().detach().cpu().numpy()[0][0]
        mask = y[0, 0, :, :].detach().cpu().numpy()
        image = X[0, 0, :, :].detach().cpu().numpy()

        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        [ax.set_axis_off() for ax in ax.flatten()]
        ax[0].imshow(image)
        ax[1].imshow(mask)
        ax[2].imshow(pred)
        ax[2].set_title(f"Dice score: {_dice_score:.3f}")
        ax[3].imshow(pred > 0.5)

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


def build_medsam_model_components(): 
    medsam_model = sam_model_registry["vit_b"](
        checkpoint="/scratch/ssd004/scratch/pwilson/medsam_vit_b_cpu.pth"
    )
    return medsam_model.image_encoder, medsam_model.prompt_encoder, medsam_model.mask_decoder


class MedSAMSegmentator(nn.Module):
    def __init__(self, image_encoder, prompt_encoder, mask_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def forward(self, image):
        image_emb = self.image_encoder(image)
        sparse_emb, dense_emb = self.prompt_encoder(None, None, None)
        mask_logits = self.mask_decoder.forward(
            image_emb,
            self.prompt_encoder.get_dense_pe(),
            sparse_emb,
            dense_emb,
            False,
        )[0]
        return mask_logits


if __name__ == "__main__":
    Experiment.submit()
