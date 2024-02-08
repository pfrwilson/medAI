import logging
import os
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass

import hydra
import medAI
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from medAI.metrics import dice_loss, dice_score
from medAI.modeling.sam import MedSAMForFinetuning, build_medsam
from medAI.utils.reproducibiliy import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)
from omegaconf import OmegaConf
from simple_parsing import choice
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

import wandb
from src.data_factory import AlignedFilesSegmentationDataFactory, BModeDataFactoryV1


@dataclass
class Args:
    fold: int | None = None
    n_folds: int | None = None
    test_center: str = "UVA"
    undersample_benign_ratio: float | None = 3.0
    min_involvement_train: float | None = 0.0
    batch_size: int = 1
    image_size: int = 1024
    augmentations: str = "translate"
    remove_benign_cores_from_positive_patients: bool = False

    optimizer: str = "adamw"
    lr: float = 1e-5
    encoder_lr: float = 1e-5
    warmup_lr: float = 1e-4
    warmup_epochs: int = 5
    wd: float = 0
    epochs: int = 30
    test_every_epoch: bool = True

    use_task_prompt: bool = False
    use_anatomical_prompt: bool = False
    use_metadata_prompt: bool = False

    base_loss: str = choice(["ce", "gce", "mae"], default="ce")
    """The base loss function that will be applied inside the valid loss region of the heatmap."""
    base_loss_prostate_mask: bool = True
    """If True, the base loss will only be applied inside the prostate mask."""
    base_loss_needle_mask: bool = True
    """If True, the base loss will only be applied inside the needle mask."""
    loss_pos_weight: float = 1.0
    """The positive class weight for the base loss function."""
    valid_loss_region_mode: tp.Literal['hard', 'soft'] = "hard"
    """The mode for defining the valid loss region. If 'hard', the valid loss pixels are selected from the valid loss region. 
    If 'soft', the valid loss region is blurred and resized to the heatmap size, and the loss is weighted by this mask."""
    use_loss_mil: bool = False
    loss_mil_loss_weight: float = 0.0
    loss_top_percentile: float = 0.2

    device: str = "cuda"
    accumulate_grad_steps: int = 8

    exp_dir: str | None = None
    checkpoint_dir: str | None = None

    project: str = "miccai2024"
    group: str | None = None
    name: str | None = None

    log_images: bool = False
    encoder_weights_path: str | None = None
    encoder_load_mode: tp.Literal['dino_medsam', 'ibot_medsam', 'image_encoder',
                                  'none'] = 'none'
    seed: int = 42
    use_amp: bool = True

    segmentation_task_prob: float = 0.5


class Experiment:

    def __init__(self, config: Args):
        self.config = config

    def setup(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logging.info("Setting up experiment")
        os.makedirs(self.config.exp_dir, exist_ok=True)
        logging.info("Running in directory: " + self.config.exp_dir)

        wandb.init(
            project=self.config.project,
            group=self.config.group,
            name=self.config.name,
            config=self.config,
        )
        logging.info("Wandb initialized")
        logging.info("Wandb url: " + wandb.run.url)

        self.exp_state_path = os.path.join(self.config.checkpoint_dir,
                                           "experiment_state.pth")
        if os.path.exists(self.exp_state_path):
            logging.info("Loading experiment state from experiment_state.pth")
            self.state = torch.load(self.exp_state_path)
        else:
            logging.info("No experiment state found - starting from scratch")
            self.state = None

        set_global_seed(self.config.seed)

        # logging setup
        self.setup_data()

        # Setup model
        self.setup_model()
        if self.state is not None:
            self.model.load_state_dict(self.state["model"])

        self.setup_optimizer()

        if self.state is not None:
            self.optimizer.load_state_dict(self.state["optimizer"])
            self.scheduler.load_state_dict(self.state["scheduler"])

        self.gradient_scaler = torch.cuda.amp.GradScaler()
        if self.state is not None:
            self.gradient_scaler.load_state_dict(self.state["gradient_scaler"])

        self.epoch = 0 if self.state is None else self.state["epoch"]
        logging.info(f"Starting at epoch {self.epoch}")
        self.best_score = 0 if self.state is None else self.state["best_score"]
        logging.info(f"Best score so far: {self.best_score}")
        if self.state is not None:
            rng_state = self.state["rng"]
            set_all_rng_states(rng_state)

    def setup_model(self):
        logging.info("Setting up model")
        self.model = MedSAMWithPCAPrompts(
            n_tasks=2,
            use_task_prompt=self.config.use_task_prompt,
            use_anatomical_prompt=self.config.use_anatomical_prompt,
            use_metadata_prompt=self.config.use_metadata_prompt,
        )
        if self.config.encoder_weights_path:
            load_encoder_weights(
                self.model.medsam_model.image_encoder,
                self.config.encoder_weights_path,
                self.config.encoder_load_mode,
            )

            # self.model.medsam_model.image_encoder.load_state_dict(
            #     torch.load(self.config.encoder_weights_path)
            # )
        self.model.to(self.config.device)
        torch.compile(self.model)
        self.model.freeze_backbone()  # freeze backbone for first few epochs

    def setup_optimizer(self):
        if self.config.wd > 0:
            raise NotImplementedError("Weight decay not implemented yet")
        match self.config.optimizer:
            case "adamw":
                opt_factory = lambda parameters, lr: optim.AdamW(
                    parameters, lr=lr, weight_decay=0)
            case "sgdw":
                opt_factory = lambda parameters, lr: optim.SGD(
                    parameters,
                    lr=lr,
                    momentum=0.9,
                    weight_decay=self.config.wd,
                )
            case _:
                raise ValueError(
                    f"Unknown optimizer: {self.config.optimizer}. Options are 'adamw' and 'sgdw'"
                )

        params = [
            {
                "params": self.model.get_encoder_parameters(),
                "lr": self.config.encoder_lr,
            },
            {
                "params": self.model.get_non_encoder_parameters(),
                "lr": self.config.lr,
            },
        ]
        self.optimizer = opt_factory(params, lr=self.config.lr)

        self.scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=5 * len(self.train_loader),
            max_epochs=self.config.epochs * len(self.train_loader),
        )

        self.warmup_optimizer = opt_factory(
            self.model.get_non_encoder_parameters(), lr=self.config.warmup_lr)

    def setup_data(self):
        logging.info("Setting up data")
        data_factory = BModeDataFactoryV1(
            fold=self.config.fold,
            n_folds=self.config.n_folds,
            test_center=self.config.test_center,
            undersample_benign_ratio=self.config.undersample_benign_ratio,
            min_involvement_train=self.config.min_involvement_train,
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            augmentations=self.config.augmentations,
            remove_benign_cores_from_positive_patients=self.config.
            remove_benign_cores_from_positive_patients,
        )
        self.train_loader = data_factory.train_loader()
        self.val_loader = data_factory.val_loader()
        self.test_loader = data_factory.test_loader()
        logging.info(f"Number of training batches: {len(self.train_loader)}")
        logging.info(f"Number of validation batches: {len(self.val_loader)}")
        logging.info(f"Number of test batches: {len(self.test_loader)}")
        logging.info(
            f"Number of training samples: {len(self.train_loader.dataset)}")
        logging.info(
            f"Number of validation samples: {len(self.val_loader.dataset)}")
        logging.info(f"Number of test samples: {len(self.test_loader.dataset)}")

        # dump core_ids to file
        train_core_ids = self.train_loader.dataset.core_ids
        val_core_ids = self.val_loader.dataset.core_ids
        test_core_ids = self.test_loader.dataset.core_ids

        with open(os.path.join(self.config.exp_dir, "train_core_ids.txt"),
                  "w") as f:
            f.write("\n".join(train_core_ids))
        with open(os.path.join(self.config.exp_dir, "val_core_ids.txt"),
                  "w") as f:
            f.write("\n".join(val_core_ids))
        with open(os.path.join(self.config.exp_dir, "test_core_ids.txt"),
                  "w") as f:
            f.write("\n".join(test_core_ids))

        wandb.save(os.path.join(self.config.exp_dir, "train_core_ids.txt"))
        wandb.save(os.path.join(self.config.exp_dir, "val_core_ids.txt"))
        wandb.save(os.path.join(self.config.exp_dir, "test_core_ids.txt"))

        data_factory_segmentation = AlignedFilesSegmentationDataFactory(
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            augmentations=self.config.augmentations,
        )
        self.train_loader_segmentation = data_factory_segmentation.train_loader(
        )
        self.val_loader_segmentation = data_factory_segmentation.val_loader()
        self.test_loader_segmentation = data_factory_segmentation.test_loader()

    def run(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.epochs):
            logging.info(f"Epoch {self.epoch}")
            self.save_experiment_state()

            self.run_train_epoch(self.train_loader, desc="train")

            val_metrics = self.run_eval_epoch(self.val_loader, desc="val")

            if val_metrics is not None:
                tracked_metric = val_metrics["val/core_auc_high_involvement"]
                new_record = tracked_metric > self.best_score
            else:
                new_record = None

            if new_record:
                self.best_score = tracked_metric
                logging.info(f"New best score: {self.best_score}")

            if new_record or self.config.test_every_epoch:
                self.training = False
                logging.info("Running test set")
                metrics = self.run_eval_epoch(self.test_loader, desc="test")
                test_score = metrics["test/core_auc_high_involvement"]
            else:
                test_score = None

            self.save_model_weights(score=test_score, is_best_score=new_record)

            # also test segmentation model
            dice_scores = []
            for i, batch in enumerate(self.test_loader_segmentation):
                dice_score = self.eval_step_segmentation(batch)
                dice_scores.append(dice_score)
            wandb.log({"test/segmentation_dice": np.mean(dice_scores)})

        logging.info("Finished training")
        self.teardown()

    def train_step_detection(self, batch):
        bmode = batch["bmode"].to(self.config.device)
        needle_mask = batch["needle_mask"].to(self.config.device)
        prostate_mask = batch["prostate_mask"].to(self.config.device)
        label = batch["label"].to(self.config.device)
        involvement = batch["involvement"].to(self.config.device)
        psa = batch["psa"].to(self.config.device)
        age = batch["age"].to(self.config.device)
        family_history = batch["family_history"].to(self.config.device)
        anatomical_location = batch["loc"].to(self.config.device)
        all_cores_benign = batch["all_cores_benign"].to(self.config.device)
        B = len(bmode)
        task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

        with torch.cuda.amp.autocast():
            heatmap_logits = self.model(
                bmode,
                task_id=task_id,
                anatomical_location=anatomical_location,
                psa=psa,
                age=age,
                family_history=family_history,
            )

            if torch.any(torch.isnan(heatmap_logits)):
                logging.warning("NaNs in heatmap logits")
                breakpoint()

            # loss = torch.tensor(0,
            #                     dtype=torch.float32,
            #                     device=heatmap_logits.device)
            # # base loss component
            # masks = []
            # for i in range(len(heatmap_logits)):
            #     mask = torch.ones(prostate_mask[i].shape,
            #                       device=prostate_mask[i].device).bool()
            #     if self.config.base_loss_prostate_mask:
            #         mask &= prostate_mask[i] > 0.5
            #     if self.config.base_loss_needle_mask:
            #         mask &= needle_mask[i] > 0.5
            #     masks.append(mask)
            # masks = torch.stack(masks)
            # predictions, batch_idx = MaskedPredictionModule()(heatmap_logits,
            #                                                   masks)
            # labels = torch.zeros(len(predictions), device=predictions.device)
            # for i in range(len(predictions)):
            #     labels[i] = label[batch_idx[i]]
            # labels = labels[..., None]  # needs to match N, C shape of preds

            #
            # if self.config.base_loss == "ce":
            #     loss += nn.functional.binary_cross_entropy_with_logits(
            #         predictions,
            #         labels,
            #         pos_weight=torch.tensor(self.config.loss_pos_weight,
            #                                 device=predictions.device),
            #     )
            # elif self.config.base_loss == "gce":
            #     # we should convert to "two class" classification problem
            #     loss_fn = BinaryGeneralizedCrossEntropy()
            #     loss += loss_fn(predictions, labels)
            # elif self.config.base_loss == "mae":
            #     loss_unreduced = nn.functional.l1_loss(predictions,
            #                                            labels,
            #                                            reduction="none")
            #     loss_unreduced[labels == 1] *= self.config.loss_pos_weight
            #     loss += loss_unreduced.mean()
            # else:
            #     raise ValueError(f"Unknown base loss: {self.config.base_loss}")
            #

            if self.config.valid_loss_region_mode == "hard":
                loss_fn = CancerDetectionValidRegionLoss(
                    base_loss=self.config.base_loss,
                    loss_pos_weight=self.config.loss_pos_weight,
                    prostate_mask=self.config.base_loss_prostate_mask,
                    needle_mask=self.config.base_loss_needle_mask,
                )
            elif self.config.valid_loss_region_mode == "soft":
                loss_fn = CancerDetectionSoftValidRegionLoss(
                    loss_pos_weight=self.config.loss_pos_weight,
                    prostate_mask=self.config.base_loss_prostate_mask,
                    needle_mask=self.config.base_loss_needle_mask,
                )
            else:
                raise ValueError(
                    f"Unknown valid loss region mode: {self.config.valid_loss_region_mode}"
                )

            loss = loss_fn(heatmap_logits, prostate_mask, needle_mask, label,
                           involvement)

            # MIL loss component outside needle
            if self.config.loss_mil_loss_weight > 0:
                masks = []
                for i in range(len(heatmap_logits)):
                    mask = prostate_mask[i] > 0.5
                    masks.append(mask)
                masks = torch.stack(masks)
                predictions, batch_idx = MaskedPredictionModule()(
                    heatmap_logits, masks)
                labels = torch.zeros(len(predictions),
                                     device=predictions.device)
                for i in range(len(predictions)):
                    labels[i] = label[batch_idx[i]]
                labels = labels[..., None]  # needs to match N, C shape of preds
                loss += self.config.loss_mil_loss_weight * simple_mil_loss(
                    predictions,
                    labels,
                    batch_idx,
                    top_percentile=self.config.loss_top_percentile,
                    pos_weight=torch.tensor(self.config.loss_pos_weight,
                                            device=predictions.device),
                )

            return loss

    def train_step_segmentation(self, batch):
        bmode = batch["bmode"].to(self.config.device)
        prostate_mask = batch["prostate_mask"].to(self.config.device)
        task_id = torch.ones(len(bmode), dtype=torch.long, device=bmode.device)

        with torch.cuda.amp.autocast():
            heatmap_logits = self.model(
                bmode,
                task_id=task_id,
            )

            loss, _dice_score = get_segmentation_loss_and_score(
                heatmap_logits, prostate_mask)

        return loss

    @torch.no_grad()
    def eval_step(self, batch, aggregate=True):
        if batch["dataset_name"][0] == "aligned_files":
            return self.eval_step_segmentation(batch)

        bmode = batch["bmode"].to(self.config.device)
        needle_mask = batch["needle_mask"].to(self.config.device)
        prostate_mask = batch["prostate_mask"].to(self.config.device)
        label = batch["label"].to(self.config.device)
        involvement = batch["involvement"].to(self.config.device)
        psa = batch["psa"].to(self.config.device)
        age = batch["age"].to(self.config.device)
        family_history = batch["family_history"].to(self.config.device)
        anatomical_location = batch["loc"].to(self.config.device)
        B = len(bmode)
        task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

        with torch.cuda.amp.autocast():
            heatmap_logits = self.model(
                bmode,
                task_id=task_id,
                anatomical_location=anatomical_location,
                psa=psa,
                age=age,
                family_history=family_history,
            )

            masks = []
            for i in range(len(heatmap_logits)):
                mask = prostate_mask[i] > 0.5
                mask = mask & (needle_mask[i] > 0.5)
                masks.append(mask)
            masks = torch.stack(masks)

            predictions, batch_idx = MaskedPredictionModule()(heatmap_logits,
                                                              masks)

            mean_predictions_in_mask = []
            for i in range(B):
                mean_predictions_in_mask.append(
                    predictions[batch_idx == i].sigmoid().mean())
            mean_predictions_in_mask = torch.stack(mean_predictions_in_mask)

            if aggregate:
                self.mean_predictions_in_mask.append(mean_predictions_in_mask)
                self.labels.append(label)
                self.involvement.append(involvement)

        return mean_predictions_in_mask, label, involvement

    @torch.no_grad()
    def eval_step_segmentation(self, batch):
        bmode = batch["bmode"].to(self.config.device)
        prostate_mask = batch["prostate_mask"].to(self.config.device)
        task_id = torch.ones(len(bmode), dtype=torch.long, device=bmode.device)

        with torch.cuda.amp.autocast():
            heatmap_logits = self.model(
                bmode,
                task_id=task_id,
            )

            loss, _dice_score = get_segmentation_loss_and_score(
                heatmap_logits, prostate_mask)

        return _dice_score.item()

    def run_train_epoch(self, loader, desc="train"):
        self.model.train()

        if self.epoch < self.config.warmup_epochs:
            optimizer = self.warmup_optimizer
            scheduler = None
        else:
            self.model.unfreeze_backbone()
            optimizer = self.optimizer
            scheduler = self.scheduler

        train_loader_segmentation_iter = iter(self.train_loader_segmentation)
        train_loader_detection_iter = iter(loader)

        i = 0
        while True:
            i += 1
            if torch.rand(1).item() < self.config.segmentation_task_prob:
                step = "segmentation"
                try:
                    batch = next(train_loader_segmentation_iter)
                except StopIteration:
                    train_loader_segmentation_iter = iter(
                        self.train_loader_segmentation)
                    batch = next(train_loader_segmentation_iter)
                loss = self.train_step_segmentation(batch)
            else:
                step = "detection"
                try:
                    batch = next(train_loader_detection_iter)
                except StopIteration:
                    break
                loss = self.train_step_detection(batch)

            if self.config.use_amp:
                self.gradient_scaler.scale(loss).backward()
            else:
                loss.backward()

            acc_steps = 1
            if i % self.config.accumulate_grad_steps == 0:
                if self.config.use_amp:
                    self.gradient_scaler.step(optimizer)
                    self.gradient_scaler.update()
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
                acc_steps = 1
            else:
                acc_steps += 1

            metrics = {f"train_loss_{step}": loss.item() / acc_steps}
            metrics["lr"] = optimizer.param_groups[0]["lr"]

            if scheduler is not None:
                scheduler.step()

            wandb.log(metrics)

            if i % 100 == 0 and self.config.log_images:
                if step == "segmentation":
                    self.show_segmentation_example(batch)
                else: 
                    self.show_example(batch)
                wandb.log({f"{desc}_{step}_example": wandb.Image(plt)})
                plt.close()

    def run_eval_epoch(self, loader, desc="eval"):
        self.model.eval()

        self.mean_predictions_in_mask = []
        self.labels = []
        self.involvement = []

        for i, batch in enumerate(tqdm(loader, desc=desc)):
            self.eval_step(batch)

            if i % 100 == 0 and self.config.log_images:
                self.show_example(batch)
                wandb.log({f"{desc}_example": wandb.Image(plt)})
                plt.close()

        self.mean_predictions_in_mask = torch.cat(self.mean_predictions_in_mask)
        self.labels = torch.cat(self.labels)
        self.involvement = torch.cat(self.involvement)

        return self.create_and_report_metrics(self.mean_predictions_in_mask,
                                              self.labels,
                                              self.involvement,
                                              desc=desc)

    def create_and_report_metrics(self,
                                  predictions,
                                  labels,
                                  involvement,
                                  desc="eval"):
        from src.utils import calculate_metrics

        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        involvement = involvement.cpu().numpy()

        core_probs = predictions
        core_labels = labels

        metrics = {}
        metrics_ = calculate_metrics(predictions,
                                     labels,
                                     log_images=self.config.log_images)
        metrics.update(metrics_)

        # high involvement core predictions
        high_involvement = involvement > 0.4
        benign = core_labels == 0
        keep = np.logical_or(high_involvement, benign)
        if keep.sum() > 0:
            core_probs = core_probs[keep]
            core_labels = core_labels[keep]
            metrics_ = calculate_metrics(core_probs,
                                         core_labels,
                                         log_images=self.config.log_images)
            metrics.update({
                f'{metric}_high_involvement': value
                for metric, value in metrics_.items()
            })

        metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = self.epoch
        wandb.log(metrics)
        return metrics

    @torch.no_grad()
    def show_example(self, batch):
        # don't log images by default, since they take up a lot of space.
        # should be considered more of a debuagging/demonstration tool
        if self.config.log_images is False:
            return

        if batch["dataset_name"][0] == "aligned_files":
            return self.show_segmentation_example(batch)

        bmode = batch["bmode"].to(self.config.device)
        needle_mask = batch["needle_mask"].to(self.config.device)
        prostate_mask = batch["prostate_mask"].to(self.config.device)
        label = batch["label"].to(self.config.device)
        involvement = batch["involvement"].to(self.config.device)
        psa = batch["psa"].to(self.config.device)
        age = batch["age"].to(self.config.device)
        family_history = batch["family_history"].to(self.config.device)
        anatomical_location = batch["loc"].to(self.config.device)
        B = len(bmode)
        task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

        logits = self.model(
            bmode,
            task_id=task_id,
            anatomical_location=anatomical_location,
            psa=psa,
            age=age,
            family_history=family_history,
        )

        mean_predictions_in_mask, label, involvement = self.eval_step(
            batch, aggregate=False)
        core_prediction = mean_predictions_in_mask[0].item()

        pred = logits.sigmoid()

        needle_mask = needle_mask.cpu()
        prostate_mask = prostate_mask.cpu()
        logits = logits.cpu()
        pred = pred.cpu()
        image = bmode.cpu()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        [ax.set_axis_off() for ax in ax.flatten()]
        kwargs = dict(vmin=0, vmax=1, extent=(0, 46, 0, 28))

        ax[0].imshow(image[0].permute(1, 2, 0), **kwargs)
        prostate_mask = prostate_mask.cpu()
        ax[0].imshow(prostate_mask[0, 0],
                     alpha=prostate_mask[0][0] * 0.3,
                     cmap="Blues",
                     **kwargs)
        ax[0].imshow(needle_mask[0, 0],
                     alpha=needle_mask[0][0],
                     cmap="Reds",
                     **kwargs)
        ax[0].set_title(f"Ground truth label: {label[0].item()}")

        ax[1].imshow(pred[0, 0], **kwargs)

        valid_loss_region = (prostate_mask[0][0]
                             > 0.5).float() * (needle_mask[0][0] > 0.5).float()

        alpha = torch.nn.functional.interpolate(valid_loss_region[None, None],
                                                size=(256, 256),
                                                mode="nearest")[0, 0]
        ax[2].imshow(pred[0, 0], alpha=alpha, **kwargs)
        ax[2].set_title(f"Core prediction: {core_prediction:.3f}")

    @torch.no_grad()
    def show_segmentation_example(self, batch):
        self.model.eval()
        X = batch["bmode"].to(self.config.device)
        y = batch["prostate_mask"].to(self.config.device)
        B = len(X)
        task_id = torch.ones(B, dtype=torch.long, device=X.device)

        mask_logits = self.model(X, task_id=task_id)
        loss, _dice_score = get_segmentation_loss_and_score(mask_logits, y)
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

    def save_experiment_state(self):
        logging.info(f"Saving experiment snapshot to {self.exp_state_path}")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
                "gradient_scaler": self.gradient_scaler.state_dict(),
                "rng": get_all_rng_states(),
            },
            self.exp_state_path,
        )

    def save_model_weights(self, score, is_best_score=False):
        if self.config.checkpoint_dir is None or not is_best_score:
            return
        logging.info("Saving model to checkpoint directory")
        logging.info(f"Checkpoint directory: {self.config.checkpoint_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.config.checkpoint_dir,
                f"best_model_epoch{self.epoch}_auc{score:.2f}.ckpt",
            ),
        )

    def teardown(self):
        # remove experiment state file
        if self.exp_state_path is not None:
            os.remove(self.exp_state_path)


class MaskedPredictionModule(nn.Module):
    """
    Computes the patch and core predictions and labels within the valid loss region for a heatmap.
    """

    def __init__(self):
        super().__init__()

    def forward(self, heatmap_logits, mask):
        """Computes the patch and core predictions and labels within the valid loss region."""
        B, C, H, W = heatmap_logits.shape

        mask = mask.float()
        mask = torch.nn.functional.interpolate(mask, size=(H, W)) > 0.5

        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[..., 0]
        logits_flattened = rearrange(heatmap_logits,
                                     "b c h w -> (b h w) c",
                                     h=H,
                                     w=W)

        logits = logits_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_logits = logits

        return patch_logits, core_idx


def involvement_tolerant_loss(patch_logits, patch_labels, core_indices,
                              involvement):
    batch_size = len(involvement)
    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)
    for i in range(batch_size):
        patch_logits_for_core = patch_logits[core_indices == i]
        patch_labels_for_core = patch_labels[core_indices == i]
        involvement_for_core = involvement[i]
        if patch_labels_for_core[0].item() == 0:
            # core is benign, so label noise is assumed to be low
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_logits_for_core, patch_labels_for_core)
        elif involvement_for_core.item() > 0.65:
            # core is high involvement, so label noise is assumed to be low
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_logits_for_core, patch_labels_for_core)
        else:
            # core is of intermediate involvement, so label noise is assumed to be high.
            # we should be tolerant of the model's "false positives" in this case.
            pred_index_sorted_by_cancer_score = torch.argsort(
                patch_logits_for_core[:, 0], descending=True)
            patch_logits_for_core = patch_logits_for_core[
                pred_index_sorted_by_cancer_score]
            patch_labels_for_core = patch_labels_for_core[
                pred_index_sorted_by_cancer_score]
            n_predictions = patch_logits_for_core.shape[0]
            patch_predictions_for_core_for_loss = patch_logits_for_core[:int(
                n_predictions * involvement_for_core.item())]
            patch_labels_for_core_for_loss = patch_labels_for_core[:int(
                n_predictions * involvement_for_core.item())]
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_predictions_for_core_for_loss,
                patch_labels_for_core_for_loss,
            )


def simple_mil_loss(
        patch_logits,
        patch_labels,
        core_indices,
        top_percentile=0.2,
        pos_weight=torch.tensor(1.0),
):
    ce_loss = nn.functional.binary_cross_entropy_with_logits(
        patch_logits, patch_labels, pos_weight=pos_weight, reduction="none")

    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)

    for i in torch.unique(core_indices):
        patch_losses_for_core = ce_loss[core_indices == i]
        n_patches = len(patch_losses_for_core)
        n_patches_to_keep = int(n_patches * top_percentile)
        patch_losses_for_core_sorted = torch.sort(patch_losses_for_core)[0]
        patch_losses_for_core_to_keep = patch_losses_for_core_sorted[:
                                                                     n_patches_to_keep]
        loss += patch_losses_for_core_to_keep.mean()

    return loss


def get_segmentation_loss_and_score(mask_logits, gt_mask):
    B, C, H, W = mask_logits.shape

    gt_mask = torch.nn.functional.interpolate(gt_mask.float(), size=(H, W))

    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        mask_logits, gt_mask)
    _dice_loss = dice_loss(mask_logits.sigmoid(), gt_mask)
    loss = ce_loss + _dice_loss

    _dice_score = dice_score(mask_logits.sigmoid(), gt_mask)
    return loss, _dice_score


class CancerDetectionValidRegionLoss(nn.Module):

    def __init__(self,
                 base_loss: str = 'ce',
                 loss_pos_weight: float = 1.0,
                 prostate_mask: bool = True,
                 needle_mask: bool = True):
        super().__init__()
        self.base_loss = base_loss
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask

    def forward(self, cancer_logits, prostate_mask, needle_mask, label,
                involvement):
        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(prostate_mask[i].shape,
                              device=prostate_mask[i].device).bool()
            if self.prostate_mask:
                mask &= prostate_mask[i] > 0.5
            if self.needle_mask:
                mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        labels = torch.zeros(len(predictions), device=predictions.device)
        for i in range(len(predictions)):
            labels[i] = label[batch_idx[i]]
        labels = labels[..., None]  # needs to match N, C shape of preds

        loss = torch.tensor(0, dtype=torch.float32, device=predictions.device)
        if self.base_loss == "ce":
            loss += nn.functional.binary_cross_entropy_with_logits(
                predictions,
                labels,
                pos_weight=torch.tensor(self.loss_pos_weight,
                                        device=predictions.device),
            )
        elif self.base_loss == "gce":
            # we should convert to "two class" classification problem
            loss_fn = BinaryGeneralizedCrossEntropy()
            loss += loss_fn(predictions, labels)
        elif self.base_loss == "mae":
            loss_unreduced = nn.functional.l1_loss(predictions,
                                                   labels,
                                                   reduction="none")
            loss_unreduced[labels == 1] *= self.config.loss_pos_weight
            loss += loss_unreduced.mean()
        else:
            raise ValueError(f"Unknown base loss: {self.config.base_loss}")

        return loss


class CancerDetectionSoftValidRegionLoss(nn.Module):

    def __init__(self,
                 loss_pos_weight: float = 1,
                 prostate_mask: bool = True,
                 needle_mask: bool = True,
                 sigma: float = 15):
        super().__init__()
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask
        self.sigma = sigma

    def forward(self, cancer_logits, prostate_mask, needle_mask, label,
                involvement):
        masks = []
        for i in range(len(cancer_logits)):
            mask = prostate_mask[i] > 0.5
            mask = mask & (needle_mask[i] > 0.5)
            mask = mask.float().cpu().numpy()[0]

            # resize and blur mask
            from skimage.transform import resize
            mask = resize(mask, (256, 256), order=0)
            from skimage.filters import gaussian
            mask = gaussian(mask, self.sigma, mode='constant', cval=0)
            mask = mask - mask.min()
            mask = mask / mask.max()
            mask = torch.tensor(mask, device=cancer_logits.device)[None, ...]

            masks.append(mask)
        masks = torch.stack(masks)

        B = label.shape[0]
        label = label.repeat(B, 1, 256, 256).float()
        loss_by_pixel = nn.functional.binary_cross_entropy_with_logits(
            cancer_logits,
            label,
            pos_weight=torch.tensor(self.loss_pos_weight,
                                    device=cancer_logits.device),
            reduction="none")
        loss = (loss_by_pixel * masks).mean()
        return loss


CORE_LOCATIONS = [
    "LML",
    "RBL",
    "LMM",
    "RMM",
    "LBL",
    "LAM",
    "RAM",
    "RML",
    "LBM",
    "RAL",
    "RBM",
    "LAL",
]


class MedSAMWithPCAPrompts(nn.Module):
    """
    Wraps the SAM model to do unprompted segmentation.

    Args:
        freeze_backbone (bool): If True, freezes the backbone of the model.
    """

    METADATA = "psa", "age", "family_history"

    def __init__(
        self,
        n_tasks=2,
        use_task_prompt=True,
        use_anatomical_prompt=True,
        use_metadata_prompt=True,
    ):
        super().__init__()
        self._frozen_backbone = False
        self.medsam_model = build_medsam()

        EMBEDDING_DIM = 256

        self.task_prompt_module = nn.Embedding(n_tasks, EMBEDDING_DIM)
        self.anatomical_prompt_module = nn.Embedding(6, EMBEDDING_DIM)
        # embed floating point values to 256 dim
        self.psa_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        self.age_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        # 3 values for family history: 0, 1, 2 (yes, no, unknown)
        self.family_history_prompt_module = nn.Embedding(3, EMBEDDING_DIM)

        self.use_task_prompt = use_task_prompt
        self.use_anatomical_prompt = use_anatomical_prompt
        self.use_metadata_prompt = use_metadata_prompt

    def forward(
        self,
        image,
        task_id=None,
        anatomical_location=None,
        psa=None,
        age=None,
        family_history=None,
    ):
        with torch.no_grad() if self._frozen_backbone else torch.enable_grad():
            image_feats = self.medsam_model.image_encoder(image)

        sparse_embedding, dense_embedding = self.medsam_model.prompt_encoder(
            None,
            None,
            None  
        )

        if self.use_task_prompt and task_id is not None:
            task_embedding = self.task_prompt_module(task_id)
            task_embedding = task_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, task_embedding],
                                         dim=1)

        if self.use_anatomical_prompt and anatomical_location is not None:
            anatomical_embedding = self.anatomical_prompt_module(
                anatomical_location)
            anatomical_embedding = anatomical_embedding[:, None, :]
            sparse_embedding = torch.cat(
                [sparse_embedding, anatomical_embedding], dim=1)

        if self.use_metadata_prompt:

            if psa is not None:
                psa_embedding = self.psa_prompt_module(psa)
                psa_embedding = psa_embedding[:, None, :]
                sparse_embedding = torch.cat([sparse_embedding, psa_embedding],
                                             dim=1)

            if age is not None:
                age_embedding = self.age_prompt_module(age)
                age_embedding = age_embedding[:, None, :]
                sparse_embedding = torch.cat([sparse_embedding, age_embedding],
                                             dim=1)

            if family_history is not None:
                family_history_embedding = self.family_history_prompt_module(
                    family_history)
                family_history_embedding = family_history_embedding[:, None, :]
                sparse_embedding = torch.cat(
                    [sparse_embedding, family_history_embedding], dim=1)

        mask_logits = self.medsam_model.mask_decoder.forward(
            image_feats,
            self.medsam_model.prompt_encoder.get_dense_pe(),
            sparse_embedding,
            dense_embedding,
            multimask_output=False,
        )[0]
        return mask_logits

    def freeze_backbone(self):
        self._frozen_backbone = True

    def unfreeze_backbone(self):
        self._frozen_backbone = False

    def get_encoder_parameters(self):
        # should separate image encoder parameters from neck parameters
        return [
            p for k, p in self.medsam_model.image_encoder.named_parameters()
            if "neck" not in k
        ]

    def get_non_encoder_parameters(self):
        from itertools import chain

        return chain(
            self.medsam_model.mask_decoder.parameters(),
            self.task_prompt_module.parameters(),
            self.anatomical_prompt_module.parameters(),
            self.psa_prompt_module.parameters(),
            self.age_prompt_module.parameters(),
            self.family_history_prompt_module.parameters(),
            self.medsam_model.image_encoder.neck.parameters(),
        )


class BinaryGeneralizedCrossEntropy(torch.nn.Module):

    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, pred, labels):
        pred = pred.sigmoid()[..., 0]
        labels = labels[..., 0].long()
        pred = torch.stack([1 - pred, pred], dim=-1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, 2).float().to(pred.device)
        gce = (1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1),
                               self.q)) / self.q
        return gce.mean()


def load_encoder_weights(image_encoder, weights_path, adapter_mode=None):
    state = torch.load(weights_path, map_location="cpu")
    if adapter_mode is None:
        image_encoder.load_state_dict(state)
    elif "dino" in adapter_mode:
        from train_medsam_dino_style import MedSAMDino

        model = MedSAMDino()
        model.load_state_dict(state)
        image_encoder_state = model.image_encoder.state_dict()
        image_encoder.load_state_dict(image_encoder_state)
    elif "ibot" in adapter_mode:
        from train_medsam_ibot_style import MedSAMIBot

        model = MedSAMIBot(8192, 8192)
        model.load_state_dict(state)
        image_encoder_state = model.image_encoder.state_dict()
        image_encoder.load_state_dict(image_encoder_state)
    else:
        raise ValueError(f"Unknown adapter mode: {adapter_mode}")


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    experiment = Experiment(args)
    experiment.run()
