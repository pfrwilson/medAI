import argparse
import logging
import os
import typing as tp
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from src.data_factory import BModeDataFactoryV1
from torch.nn import functional as F
from tqdm import tqdm

from medAI.utils.reproducibiliy import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)


def parse_args():
    # fmt: off

    from rich_argparse import ArgumentDefaultsRichHelpFormatter
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsRichHelpFormatter)

    group = parser.add_argument_group("Data", "Arguments related to data loading and preprocessing")
    group.add_argument("--fold", type=int, default=None, help="The fold to use. If not specified, uses leave-one-center-out cross-validation.")
    group.add_argument("--n_folds", type=int, default=None, help="The number of folds to use for cross-validation.")
    group.add_argument("--test_center", type=str, default=None, 
                        help="If not None, uses leave-one-center-out cross-validation with the specified center as test. If None, uses k-fold cross-validation.")
    group.add_argument("--val_seed", type=int, default=0, 
                       help="The seed to use for validation split.")            
    group.add_argument("--undersample_benign_ratio", type=lambda x: float(x) if not x.lower() == 'none' else None, default=None,
                       help="""If not None, undersamples benign cores with the specified ratio.""")
    group.add_argument("--min_involvement_train", type=float, default=0.0,
                       help="""The minimum involvement threshold to use for training.""")
    group.add_argument("--batch_size", type=int, default=1, help="The batch size to use for training.")
    group.add_argument("--augmentations", type=str, default="translate", help="The augmentations to use for training.")
    group.add_argument("--remove_benign_cores_from_positive_patients", action="store_true", help="If True, removes benign cores from positive patients (training only).")
    group.add_argument("--image_size", type=int, default=1024, help="The size to use for the images.")
    group.add_argument("--mask_size", type=int, default=256, help="The size to use for the masks.")
    group.add_argument("--limit_train_data", type=float, default=1., 
                       help="""If less than 1, chooses a center-balanced subset of the original train data to train with. The value given is the fraction of the original data to use.""")
    group.add_argument("--train_subsample_seed", type=int, default=42, help="The seed to use for subsampling the training data (if limit_train_data < 1).") 
    group = parser.add_argument_group("Training", "Arguments related to training.")
    group.add_argument("--optimizer", type=str, default="adamw", help="The optimizer to use for training.")
    group.add_argument("--lr", type=float, default=1e-5, help="LR for the model the mask decoder and prompt encoder portions of the model during Stage 2.")
    group.add_argument("--encoder_lr", type=float, default=1e-5, help="LR for the encoder part of the model during Stage 2.")
    group.add_argument("--warmup_lr", type=float, default=1e-4, help="LR for the warmup, frozen encoder part.")
    group.add_argument("--cnn_lr", type=float, default=1e-5, help="LR for the CNN part of the model.")
    group.add_argument("--warmup_epochs", type=int, default=5, help="The number of epochs to train the warmup, frozen encoder part.")
    group.add_argument("--wd", type=float, default=0, help="The weight decay to use for training.")
    group.add_argument("--epochs", type=int, default=30, help="The number of epochs to train for in terms of LR scheduler.")
    group.add_argument("--cutoff_epoch", type=int, default=None, help="If not None, the training will stop after this epoch, but this will not affect the learning rate scheduler.")
    group.add_argument("--test_every_epoch", action="store_true", help="If True, runs the test set every epoch.")
    group.add_argument("--accumulate_grad_steps", type=int, default=8, help="The number of gradient accumulation steps to use.")

    # MODEL
    ProFound.add_arguments(parser)

    # LOSS
    parser.add_argument("--n_loss_terms", type=int, default=1, help="""The number of loss terms to use.
                        Although we found no benefit beyond using a single masked CE loss, the code supports multiple loss terms.""")
    args, _ = parser.parse_known_args()
    n_loss_terms = args.n_loss_terms
    for i in range(n_loss_terms):
        group = parser.add_argument_group(f"Loss term {i}", f"Arguments related to loss term {i}.")
        group.add_argument(f"--loss_{i}_name", type=str, default="valid_region", choices=('valid_region',), help="The name of the loss function to use."),
        group.add_argument(f"--loss_{i}_base_loss_name", type=str, default="ce", 
                           choices=('ce', 'gce', 'mae', 'mil'), 
                           help="""The name of the lower-level loss function to use. Our experiments showed best performance with simple CE loss, but 
                           due to weak supervision and label noise, we also experimented with GCE, MAE, and MIL. However, we found no benefit beyond using CE loss.""")
        def str2bool(str): 
            return True if str.lower() == 'true' else False
        group.add_argument(f"--loss_{i}_pos_weight", type=float, default=1.0, help="""The positive class weight for the loss function. If using a large ratio of benign to
                           cancer cores in training, it is recommended to increase this value to 2 or 3 to account for the class imbalance.""")
        group.add_argument(f"--loss_{i}_prostate_mask", type=str2bool, default=True, help="If True, the loss will only be applied inside the prostate mask.")
        group.add_argument(f"--loss_{i}_needle_mask", type=str2bool, default=True, help="If True, the loss will only be applied inside the needle mask.")
        group.add_argument(f"--loss_{i}_weight", type=float, default=1.0, help="The weight to use for the loss function.")

    # WANDB
    group = parser.add_argument_group("Wandb", "Arguments related to wandb.")
    group.add_argument("--project", type=str, default="miccai2024", help="The wandb project to use.")
    group.add_argument("--group", type=str, default=None, help="The wandb group to use.")
    group.add_argument("--name", type=str, default=None, help="The wandb name to use.")
    group.add_argument("--log_images", action="store_true", help="If True, logs images to wandb.")
    group.add_argument("--tags", type=str, nargs="+", default=[], help="The tags to use for wandb.")

    group = parser.add_argument_group("Misc", "Miscellaneous arguments.")
    group.add_argument("--encoder_weights_path", type=str, default=None, help="The path to the encoder weights to use. If None, uses the Foundation Model initialization")
    group.add_argument("--encoder_load_mode", type=str, default="none", choices=("dino_medsam", "ibot_medsam", "image_encoder", "none"), help="The mode to use for loading the encoder weights.")
    group.add_argument("--seed", type=int, default=42, help="The seed to use for training.")
    group.add_argument("--use_amp", action="store_true", help="If True, uses automatic mixed precision.")
    group.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="The device to use for training")
    group.add_argument("--exp_dir", type=str, default="experiments/default", help="The directory to use for the experiment.")
    group.add_argument("--checkpoint_dir", type=str, default=None, help="The directory to use for the checkpoints. If None, does not save checkpoints.")
    group.add_argument("--debug", action="store_true", help="If True, runs in debug mode.")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit')

    args = parser.parse_args()
    return args
    # fmt: on


class Experiment:
    def __init__(self, config):
        self.config = config

    def setup(self):
        logging.basicConfig(
            level=logging.INFO if not self.config.debug else logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logging.info("Setting up experiment")
        os.makedirs(self.config.exp_dir, exist_ok=True)
        logging.info("Running in directory: " + self.config.exp_dir)

        if self.config.debug:
            self.config.name = "debug"
        wandb.init(
            project=self.config.project,
            group=self.config.group,
            name=self.config.name,
            config=self.config,
            tags=self.config.tags,
        )
        logging.info("Wandb initialized")
        logging.info("Wandb url: " + wandb.run.url)

        if self.config.checkpoint_dir is not None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            self.exp_state_path = os.path.join(
                self.config.checkpoint_dir, "experiment_state.pth"
            )
            if os.path.exists(self.exp_state_path):
                logging.info("Loading experiment state from experiment_state.pth")
                self.state = torch.load(self.exp_state_path)
            else:
                logging.info("No experiment state found - starting from scratch")
                self.state = None
        else:
            self.exp_state_path = None
            self.state = None

        set_global_seed(self.config.seed)

        self.setup_data()

        self.setup_model()
        if self.state is not None:
            self.model.load_state_dict(self.state["model"])

        self.setup_optimizer()
        if self.state is not None:
            self.optimizer.load_state_dict(self.state["optimizer"])
            self.lr_scheduler.load_state_dict(self.state["lr_scheduler"])

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

        self.model = ProFound.from_args(self.config)

        self.model.to(self.config.device)
        torch.compile(self.model)

        logging.info("Model setup complete")
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        logging.info(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        # setup criterion
        loss_terms = []
        loss_weights = []
        for i in range(self.config.n_loss_terms):
            loss_name = getattr(self.config, f"loss_{i}_name")
            base_loss_name = getattr(self.config, f"loss_{i}_base_loss_name")
            loss_pos_weight = getattr(self.config, f"loss_{i}_pos_weight")
            loss_prostate_mask = getattr(self.config, f"loss_{i}_prostate_mask")
            loss_needle_mask = getattr(self.config, f"loss_{i}_needle_mask")
            loss_weight = getattr(self.config, f"loss_{i}_weight")

            if loss_name == "valid_region":
                loss_terms.append(
                    CancerDetectionValidRegionLoss(
                        base_loss=base_loss_name,
                        loss_pos_weight=loss_pos_weight,
                        prostate_mask=loss_prostate_mask,
                        needle_mask=loss_needle_mask,
                    )
                )
                loss_weights.append(loss_weight)
            else:
                raise ValueError(f"Unknown loss name: {loss_name}")

        self.loss_fn = MultiTermCanDetLoss(loss_terms, loss_weights)

    def setup_optimizer(self):
        from torch.optim import AdamW

        (
            encoder_parameters,
            warmup_parameters,
            cnn_parameters,
        ) = self.model.get_params_groups()

        total_epochs = self.config.epochs
        encoder_frozen_epochs = self.config.warmup_epochs
        warmup_epochs = 5
        niter_per_ep = len(self.train_loader)
        warmup_lr_factor = self.config.warmup_lr / self.config.lr
        params = [
            {"params": encoder_parameters, "lr": self.config.encoder_lr},
            {"params": warmup_parameters, "lr": self.config.lr},
            {"params": cnn_parameters, "lr": self.config.cnn_lr},
        ]

        def compute_lr_multiplier(iter, is_encoder_or_cnn=True):
            if iter < encoder_frozen_epochs * niter_per_ep:
                if is_encoder_or_cnn:
                    return 0
                else:
                    if iter < warmup_epochs * niter_per_ep:
                        return (iter * warmup_lr_factor) / (
                            warmup_epochs * niter_per_ep
                        )
                    else:
                        cur_iter_in_frozen_phase = iter - warmup_epochs * niter_per_ep
                        total_iter_in_frozen_phase = (
                            encoder_frozen_epochs - warmup_epochs
                        ) * niter_per_ep
                        return (
                            0.5
                            * (
                                1
                                + np.cos(
                                    np.pi
                                    * cur_iter_in_frozen_phase
                                    / (total_iter_in_frozen_phase)
                                )
                            )
                            * warmup_lr_factor
                        )
            else:
                iter -= encoder_frozen_epochs * niter_per_ep
                if iter < warmup_epochs * niter_per_ep:
                    return iter / (warmup_epochs * niter_per_ep)
                else:
                    cur_iter = iter - warmup_epochs * niter_per_ep
                    total_iter = (
                        total_epochs - warmup_epochs - encoder_frozen_epochs
                    ) * niter_per_ep
                    return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))

        self.optimizer = AdamW(params, lr=self.config.lr, weight_decay=self.config.wd)
        from torch.optim.lr_scheduler import LambdaLR

        self.lr_scheduler = LambdaLR(
            self.optimizer,
            [
                lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True),
                lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=False),
                lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True),
            ],
        )

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
            mask_size=self.config.mask_size,
            augmentations=self.config.augmentations,
            remove_benign_cores_from_positive_patients=self.config.remove_benign_cores_from_positive_patients,
            val_seed=self.config.val_seed,
            limit_train_data=self.config.limit_train_data
            if self.config.limit_train_data < 1
            else None,
            train_subset_seed=self.config.train_subsample_seed,
        )
        self.train_loader = data_factory.train_loader()
        self.val_loader = data_factory.val_loader()
        self.test_loader = data_factory.test_loader()
        logging.info(f"Number of training batches: {len(self.train_loader)}")
        logging.info(f"Number of validation batches: {len(self.val_loader)}")
        logging.info(f"Number of test batches: {len(self.test_loader)}")
        logging.info(f"Number of training samples: {len(self.train_loader.dataset)}")
        logging.info(f"Number of validation samples: {len(self.val_loader.dataset)}")
        logging.info(f"Number of test samples: {len(self.test_loader.dataset)}")

        # dump core_ids to file
        train_core_ids = self.train_loader.dataset.core_ids
        val_core_ids = self.val_loader.dataset.core_ids
        test_core_ids = self.test_loader.dataset.core_ids

        with open(os.path.join(self.config.exp_dir, "train_core_ids.txt"), "w") as f:
            f.write("\n".join(train_core_ids))
        with open(os.path.join(self.config.exp_dir, "val_core_ids.txt"), "w") as f:
            f.write("\n".join(val_core_ids))
        with open(os.path.join(self.config.exp_dir, "test_core_ids.txt"), "w") as f:
            f.write("\n".join(test_core_ids))

        wandb.save(os.path.join(self.config.exp_dir, "train_core_ids.txt"))
        wandb.save(os.path.join(self.config.exp_dir, "val_core_ids.txt"))
        wandb.save(os.path.join(self.config.exp_dir, "test_core_ids.txt"))

    def run(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.epochs):
            if (
                self.config.cutoff_epoch is not None
                and self.epoch > self.config.cutoff_epoch
            ):
                break
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

        logging.info("Finished training")
        self.teardown()

    def run_train_epoch(self, loader, desc="train"):
        # setup epoch
        self.model.train()
        from medAI.utils.accumulators import DataFrameCollector

        accumulator = DataFrameCollector()

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            batch_for_image_generation = (
                batch.copy()
            )  # we pop some keys from batch, so we keep a copy for image generation
            if self.config.debug and train_iter > 10:
                break

            # extracting relevant data from the batch
            bmode = batch.pop("bmode").to(self.config.device)
            needle_mask = batch.pop("needle_mask").to(self.config.device)
            prostate_mask = batch.pop("prostate_mask").to(self.config.device)

            psa = batch["psa"].to(self.config.device)
            age = batch["age"].to(self.config.device)
            label = batch["label"].to(self.config.device)
            involvement = batch["involvement"].to(self.config.device)
            family_history = batch["family_history"].to(self.config.device)
            anatomical_location = batch["loc"].to(self.config.device)
            B = len(bmode)
            task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

            # run the model
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass
                heatmap_logits = self.model(
                    bmode,
                    task_id=task_id,
                    anatomical_location=anatomical_location,
                    psa=psa,
                    age=age,
                    family_history=family_history,
                    prostate_mask=prostate_mask,
                    needle_mask=needle_mask,
                )

                if torch.any(torch.isnan(heatmap_logits)):
                    logging.warning("NaNs in heatmap logits")
                    breakpoint()

                # loss calculation
                loss = self.loss_fn(
                    heatmap_logits, prostate_mask, needle_mask, label, involvement
                )

                # compute predictions
                masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
                predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)
                mean_predictions_in_needle = []
                for j in range(B):
                    mean_predictions_in_needle.append(
                        predictions[batch_idx == j].sigmoid().mean()
                    )
                mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)

                prostate_masks = prostate_mask > 0.5
                predictions, batch_idx = MaskedPredictionModule()(
                    heatmap_logits, prostate_masks
                )
                mean_predictions_in_prostate = []
                for j in range(B):
                    mean_predictions_in_prostate.append(
                        predictions[batch_idx == j].sigmoid().mean()
                    )
                mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)

            loss = loss / self.config.accumulate_grad_steps
            # backward pass
            if self.config.use_amp:
                logging.debug("Backward pass")
                self.gradient_scaler.scale(loss).backward()
            else:
                logging.debug("Backward pass")
                loss.backward()

            # gradient accumulation and optimizer step
            if self.config.debug:
                for param in self.optimizer.param_groups[1]["params"]:
                    break
                logging.debug(param.data.view(-1)[0])

            if (train_iter + 1) % self.config.accumulate_grad_steps == 0:
                logging.debug("Optimizer step")
                if self.config.use_amp:
                    self.gradient_scaler.step(self.optimizer)
                    self.gradient_scaler.update()
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.config.debug:
                    for param in self.optimizer.param_groups[1]["params"]:
                        break
                    logging.debug(param.data.view(-1)[0])

            self.lr_scheduler.step()

            # accumulate outputs
            accumulator(
                {
                    "average_needle_heatmap_value": mean_predictions_in_needle,
                    "average_prostate_heatmap_value": mean_predictions_in_prostate,
                    **batch,
                }
            )

            # log metrics
            step_metrics = {
                "train_loss": loss.item() / self.config.accumulate_grad_steps
            }
            encoder_lr = self.optimizer.param_groups[0]["lr"]
            main_lr = self.optimizer.param_groups[1]["lr"]
            cnn_lr = self.optimizer.param_groups[2]["lr"]
            step_metrics["encoder_lr"] = encoder_lr
            step_metrics["main_lr"] = main_lr
            step_metrics["cnn_lr"] = cnn_lr

            wandb.log(step_metrics)

            # log images
            if train_iter % 100 == 0 and self.config.log_images:
                self.show_example(batch_for_image_generation)
                wandb.log({f"{desc}_example": wandb.Image(plt)})
                plt.close()

        # compute and log metrics
        results_table = accumulator.compute()
        # results_table.to_csv(os.path.join(self.config.exp_dir, f"{desc}_epoch_{self.epoch}_results.csv"))
        # wandb.save(os.path.join(self.config.exp_dir, f"{desc}_epoch_{self.epoch}_results.csv"))
        return self.create_and_report_metrics(results_table, desc="train")

    @torch.no_grad()
    def run_eval_epoch(self, loader, desc="eval"):
        self.model.eval()

        from medAI.utils.accumulators import DataFrameCollector

        accumulator = DataFrameCollector()

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            batch_for_image_generation = (
                batch.copy()
            )  # we pop some keys from batch, so we keep a copy for image generation
            bmode = batch.pop("bmode").to(self.config.device)
            needle_mask = batch.pop("needle_mask").to(self.config.device)
            prostate_mask = batch.pop("prostate_mask").to(self.config.device)

            psa = batch["psa"].to(self.config.device)
            age = batch["age"].to(self.config.device)
            family_history = batch["family_history"].to(self.config.device)
            anatomical_location = batch["loc"].to(self.config.device)
            B = len(bmode)
            task_id = torch.zeros(B, dtype=torch.long, device=bmode.device)

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                heatmap_logits = self.model(
                    bmode,
                    task_id=task_id,
                    anatomical_location=anatomical_location,
                    psa=psa,
                    age=age,
                    family_history=family_history,
                    prostate_mask=prostate_mask,
                    needle_mask=needle_mask,
                )

                # compute predictions
                masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
                predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)
                mean_predictions_in_needle = []
                for j in range(B):
                    mean_predictions_in_needle.append(
                        predictions[batch_idx == j].sigmoid().mean()
                    )
                mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)

                prostate_masks = prostate_mask > 0.5
                predictions, batch_idx = MaskedPredictionModule()(
                    heatmap_logits, prostate_masks
                )
                mean_predictions_in_prostate = []
                for j in range(B):
                    mean_predictions_in_prostate.append(
                        predictions[batch_idx == j].sigmoid().mean()
                    )
                mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)

            if train_iter % 100 == 0 and self.config.log_images:
                self.show_example(batch_for_image_generation)
                wandb.log({f"{desc}_example": wandb.Image(plt)})
                plt.close()

            accumulator(
                {
                    "average_needle_heatmap_value": mean_predictions_in_needle,
                    "average_prostate_heatmap_value": mean_predictions_in_prostate,
                    **batch,
                }
            )

        results_table = accumulator.compute()

        return self.create_and_report_metrics(results_table, desc=desc)

    def create_and_report_metrics(self, results_table, desc="eval"):
        from src.utils import calculate_metrics

        # core predictions
        predictions = results_table.average_needle_heatmap_value.values
        labels = results_table.label.values
        involvement = results_table.involvement.values

        core_probs = predictions
        core_labels = labels

        metrics = {}
        metrics_ = calculate_metrics(
            predictions, labels, log_images=self.config.log_images
        )
        metrics.update(metrics_)

        # high involvement core predictions
        high_involvement = involvement > 0.4
        benign = core_labels == 0
        keep = np.logical_or(high_involvement, benign)
        if keep.sum() > 0:
            core_probs = core_probs[keep]
            core_labels = core_labels[keep]
            metrics_ = calculate_metrics(
                core_probs, core_labels, log_images=self.config.log_images
            )
            metrics.update(
                {
                    f"{metric}_high_involvement": value
                    for metric, value in metrics_.items()
                }
            )

        # patient predictions
        predictions = (
            results_table.groupby("patient_id")
            .average_prostate_heatmap_value.mean()
            .values
        )
        labels = (
            results_table.groupby("patient_id").clinically_significant.sum() > 0
        ).values
        metrics_ = calculate_metrics(
            predictions, labels, log_images=self.config.log_images
        )
        metrics.update(
            {f"{metric}_patient": value for metric, value in metrics_.items()}
        )

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
            prostate_mask=prostate_mask,
            needle_mask=needle_mask,
        )

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
        ax[0].imshow(
            prostate_mask[0, 0], alpha=prostate_mask[0][0] * 0.3, cmap="Blues", **kwargs
        )
        ax[0].imshow(needle_mask[0, 0], alpha=needle_mask[0][0], cmap="Reds", **kwargs)
        ax[0].set_title(f"Ground truth label: {label[0].item()}")

        ax[1].imshow(pred[0, 0], **kwargs)

        valid_loss_region = (prostate_mask[0][0] > 0.5).float() * (
            needle_mask[0][0] > 0.5
        ).float()

        alpha = torch.nn.functional.interpolate(
            valid_loss_region[None, None],
            size=(self.config.mask_size, self.config.mask_size),
            mode="nearest",
        )[0, 0]
        ax[2].imshow(pred[0, 0], alpha=alpha, **kwargs)

    def save_experiment_state(self):
        if self.exp_state_path is None:
            return
        logging.info(f"Saving experiment snapshot to {self.exp_state_path}")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
                "gradient_scaler": self.gradient_scaler.state_dict(),
                "rng": get_all_rng_states(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            },
            self.exp_state_path,
        )

    def save_model_weights(self, score, is_best_score=False):
        if self.config.checkpoint_dir is None:
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
        # save config as json file
        with open(
            os.path.join(
                self.config.checkpoint_dir,
                f"config.json",
            ),
            "w",
        ) as f:
            import json

            json.dump(vars(self.config), f)

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

        assert mask.shape == (
            B,
            1,
            H,
            W,
        ), f"Expected mask shape to be {(B, 1, H, W)}, got {mask.shape} instead."

        # mask = mask.float()
        # mask = torch.nn.functional.interpolate(mask, size=(H, W)) > 0.5

        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[..., 0]
        logits_flattened = rearrange(heatmap_logits, "b c h w -> (b h w) c", h=H, w=W)

        logits = logits_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_logits = logits

        return patch_logits, core_idx


def involvement_tolerant_loss(patch_logits, patch_labels, core_indices, involvement):
    batch_size = len(involvement)
    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)
    for i in range(batch_size):
        patch_logits_for_core = patch_logits[core_indices == i]
        patch_labels_for_core = patch_labels[core_indices == i]
        involvement_for_core = involvement[i]
        if patch_labels_for_core[0].item() == 0:
            # core is benign, so label noise is assumed to be low
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_logits_for_core, patch_labels_for_core
            )
        elif involvement_for_core.item() > 0.65:
            # core is high involvement, so label noise is assumed to be low
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_logits_for_core, patch_labels_for_core
            )
        else:
            # core is of intermediate involvement, so label noise is assumed to be high.
            # we should be tolerant of the model's "false positives" in this case.
            pred_index_sorted_by_cancer_score = torch.argsort(
                patch_logits_for_core[:, 0], descending=True
            )
            patch_logits_for_core = patch_logits_for_core[
                pred_index_sorted_by_cancer_score
            ]
            patch_labels_for_core = patch_labels_for_core[
                pred_index_sorted_by_cancer_score
            ]
            n_predictions = patch_logits_for_core.shape[0]
            patch_predictions_for_core_for_loss = patch_logits_for_core[
                : int(n_predictions * involvement_for_core.item())
            ]
            patch_labels_for_core_for_loss = patch_labels_for_core[
                : int(n_predictions * involvement_for_core.item())
            ]
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
        patch_logits, patch_labels, pos_weight=pos_weight, reduction="none"
    )

    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)

    for i in torch.unique(core_indices):
        patch_losses_for_core = ce_loss[core_indices == i]
        n_patches = len(patch_losses_for_core)
        n_patches_to_keep = int(n_patches * top_percentile)
        patch_losses_for_core_sorted = torch.sort(patch_losses_for_core)[0]
        patch_losses_for_core_to_keep = patch_losses_for_core_sorted[:n_patches_to_keep]
        loss += patch_losses_for_core_to_keep.mean()

    return loss


class CancerDetectionLossBase(nn.Module):
    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        raise NotImplementedError


class CancerDetectionValidRegionLoss(CancerDetectionLossBase):
    def __init__(
        self,
        base_loss: str = "ce",
        loss_pos_weight: float = 1.0,
        prostate_mask: bool = True,
        needle_mask: bool = True,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask

    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
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
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=predictions.device
                ),
            )
        elif self.base_loss == "gce":
            # we should convert to "two class" classification problem
            loss_fn = BinaryGeneralizedCrossEntropy()
            loss += loss_fn(predictions, labels)
        elif self.base_loss == "mae":
            loss_unreduced = nn.functional.l1_loss(
                predictions, labels, reduction="none"
            )
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()
        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")

        return loss


class CancerDetectionSoftValidRegionLoss(CancerDetectionLossBase):
    def __init__(
        self,
        loss_pos_weight: float = 1,
        prostate_mask: bool = True,
        needle_mask: bool = True,
        sigma: float = 15,
    ):
        super().__init__()
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask
        self.sigma = sigma

    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        masks = []
        for i in range(len(cancer_logits)):
            mask = prostate_mask[i] > 0.5
            mask = mask & (needle_mask[i] > 0.5)
            mask = mask.float().cpu().numpy()[0]

            # resize and blur mask
            from skimage.transform import resize

            mask = resize(mask, (256, 256), order=0)
            from skimage.filters import gaussian

            mask = gaussian(mask, self.sigma, mode="constant", cval=0)
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
            pos_weight=torch.tensor(self.loss_pos_weight, device=cancer_logits.device),
            reduction="none",
        )
        loss = (loss_by_pixel * masks).mean()
        return loss


class MultiTermCanDetLoss(CancerDetectionLossBase):
    def __init__(self, loss_terms: list[CancerDetectionLossBase], weights: list[float]):
        super().__init__()
        self.loss_terms = loss_terms
        self.weights = weights

    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        loss = torch.tensor(0, dtype=torch.float32, device=cancer_logits.device)
        for term, weight in zip(self.loss_terms, self.weights):
            loss += weight * term(
                cancer_logits, prostate_mask, needle_mask, label, involvement
            )
        return loss


class ProFound(nn.Module):
    PROMPT_OPTIONS = [
        "task",
        "anatomical",
        "psa",
        "age",
        "family_history",
        "prostate_mask",
        "sparse_cnn_patch_features",
        "data_independent_prompts",
    ]

    BACKBONE_OPTIONS = [
        "sam",
        "medsam",
        "sam_med2d",
        "adapter_medsam",
    ]

    def __init__(
        self,
        n_tasks=1,
        prompts: list[str] = [],
        prompt_dropout: float = 0.0,  # dropout for prompt embeddings
        sam_backbone: tp.Literal[
            "sam", "medsam", "sam_med2d", "adapter_medsam"
        ] = "medsam",
        replace_patch_embed: bool = False,
        sparse_cnn_backbone_path: str = None,
        freeze_mask_decoder: bool = False,
        freeze_image_encoder: bool = False,
        freeze_cnn: bool = False,
    ):
        super().__init__()
        self.prompts = prompts
        self.prompt_dropout = prompt_dropout
        self.replace_patch_embed = replace_patch_embed
        if replace_patch_embed and sam_backbone != "sam_med2d":
            raise ValueError(
                "replace_patch_embed is only supported for sam_med2d backbone"
            )

        self.sparse_cnn_backbone_path = sparse_cnn_backbone_path

        for p in prompts:
            if not p in self.PROMPT_OPTIONS:
                raise ValueError(
                    f"Unknown prompt option: {p}. Options are {self.PROMPT_OPTIONS}"
                )

        from medAI.modeling.sam import (
            build_adapter_medsam_256,
            build_adapter_sam,
            build_adapter_sammed_2d,
            build_medsam,
            build_sam,
            build_sammed_2d,
        )

        # BUILD BACKBONE
        if sam_backbone == "medsam":
            self.medsam_model = build_medsam()
            self.image_size_for_features = 1024
        elif sam_backbone == "adapter_medsam":
            self.medsam_model = build_adapter_medsam_256()
            self.image_size_for_features = 1024
        elif sam_backbone == "sam":
            self.medsam_model = build_sam()
            self.image_size_for_features = 1024
        elif sam_backbone == "adapter_sam":
            self.medsam_model = build_adapter_sam()
            self.image_size_for_features = 1024
        elif sam_backbone == "sam_med2d":
            self.medsam_model = build_sammed_2d()

            if replace_patch_embed:
                self.image_size_for_features = 1024
                # sammed_2d has a different input size. Let's hack the model to accept 1024x1024 images
                from einops.layers.torch import Rearrange

                new_patch_embed = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.GroupNorm(32, 64),
                    nn.GELU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(32, 64),
                    nn.GELU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(32, 64),
                    nn.GELU(),
                    nn.Conv2d(64, 768, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(32, 768),
                    nn.GELU(),
                    nn.MaxPool2d(4, 4),
                    Rearrange("b c h w -> b h w c"),
                )
                self.medsam_model.image_encoder.patch_embed = new_patch_embed
            else:
                # use the default patch embed which is designed for 256x256 images
                self.image_size_for_features = 256
        elif sam_backbone == "adapter_sammed_2d":
            self.medsam_model = build_adapter_sammed_2d()
            self.image_size_for_features = 256

        if freeze_image_encoder:
            logging.debug("Freezing image encoder")
            for param in self.medsam_model.image_encoder.parameters():
                param.requires_grad = False

        if freeze_mask_decoder:
            logging.debug("Freezing mask decoder")
            for param in self.medsam_model.mask_decoder.parameters():
                param.requires_grad = False

        # BUILD PROMPT MODULES
        EMBEDDING_DIM = 256

        # null prompt - used for prompt dropout
        self.null_prompt = nn.Parameter(torch.zeros(1, EMBEDDING_DIM))

        # used for multitask training, but not currently used
        self.task_prompt_module = nn.Embedding(n_tasks, EMBEDDING_DIM)

        # 6 anatomical locations (mid-lateral, mid-medial, apex-lateral, apex-medial, base-lateral, base-medial)
        self.anatomical_prompt_module = nn.Embedding(6, EMBEDDING_DIM)

        # embed floating point values to 256 dim
        self.psa_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )

        # embed floating point values to 256 dim
        self.age_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )

        # 3 values for family history: 0, 1, 2 (yes, no, unknown)
        self.family_history_prompt_module = nn.Embedding(3, EMBEDDING_DIM)

        # CNN for extracting patch features
        from timm.models.resnet import resnet10t

        model = resnet10t(
            in_chans=3,
        )
        model.fc = nn.Identity()
        model = nn.Sequential(nn.InstanceNorm2d(3), model)
        if sparse_cnn_backbone_path is not None:
            state = torch.load(sparse_cnn_backbone_path, map_location="cpu")
            model.load_state_dict(
                {
                    k.replace("backbone.", ""): v
                    for k, v in state.items()
                    if "backbone" in k
                }
            )
        self.patch_feature_cnn = model
        if freeze_cnn:
            for param in self.patch_feature_cnn.parameters():
                param.requires_grad = False

        # project the CNN features to the prompt space
        self.patch_feature_prompt_module = nn.Linear(512, EMBEDDING_DIM)
        self.pad_token = nn.Parameter(
            torch.zeros(EMBEDDING_DIM)
        )  # for padding the number of patches to a fixed number in a minibatch

        # data independent prompts
        self.data_independent_prompts = nn.Parameter(torch.randn(1, 10, EMBEDDING_DIM))

    def forward(
        self,
        image=None,
        task_id=None,
        anatomical_location=None,
        psa=None,
        age=None,
        family_history=None,
        prostate_mask=None,
        needle_mask=None,
    ):
        DEVICE = image.device
        B, C, H, W = image.shape
        if H != self.image_size_for_features or W != self.image_size_for_features:
            image_resized_for_features = torch.nn.functional.interpolate(
                image, size=(self.image_size_for_features, self.image_size_for_features)
            )
        else:
            image_resized_for_features = image
        image_feats = self.medsam_model.image_encoder(image_resized_for_features)

        if "prostate_mask" in self.prompts:
            if (
                prostate_mask is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                mask = None
            else:
                B, C, H, W = prostate_mask.shape
                if H != 256 or W != 256:
                    prostate_mask = torch.nn.functional.interpolate(
                        prostate_mask, size=(256, 256)
                    )
                mask = prostate_mask
        else:
            mask = None

        sparse_embedding, dense_embedding = self.medsam_model.prompt_encoder.forward(
            None, None, mask
        )
        sparse_embedding = sparse_embedding.repeat_interleave(len(image), 0)

        if "task" in self.prompts:
            task_embedding = self.task_prompt_module(task_id)
            task_embedding = task_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, task_embedding], dim=1)

        if "anatomical" in self.prompts:
            if (
                anatomical_location is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                anatomical_embedding = self.null_prompt.repeat_interleave(
                    len(task_id), 0
                )
            else:
                anatomical_embedding = self.anatomical_prompt_module(
                    anatomical_location
                )
            anatomical_embedding = anatomical_embedding[:, None, :]
            sparse_embedding = torch.cat(
                [sparse_embedding, anatomical_embedding], dim=1
            )

        if "psa" in self.prompts:
            if (
                psa is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                psa_embedding = self.null_prompt.repeat_interleave(len(task_id), 0)
            else:
                psa_embedding = self.psa_prompt_module(psa)
            psa_embedding = psa_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, psa_embedding], dim=1)

        if "age" in self.prompts:
            if (
                age is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                age_embedding = self.null_prompt.repeat_interleave(len(task_id), 0)
            else:
                age_embedding = self.age_prompt_module(age)
            age_embedding = age_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, age_embedding], dim=1)

        if "family_history" in self.prompts:
            if (
                family_history is None
                or self.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.prompt_dropout
            ):
                family_history = torch.ones_like(task_id) * 2  # this encodes "unknown"
            family_history_embedding = self.family_history_prompt_module(family_history)
            family_history_embedding = family_history_embedding[:, None, :]
            sparse_embedding = torch.cat(
                [sparse_embedding, family_history_embedding], dim=1
            )

        if "data_independent_prompts" in self.prompts:
            sparse_embedding = torch.cat(
                [
                    sparse_embedding,
                    self.data_independent_prompts.repeat_interleave(B, 0),
                ],
                dim=1,
            )

        if "sparse_cnn_patch_features" in self.prompts:
            # we need to extract patches from the images.
            patches = []
            batch_indices = []
            positions = []

            B = len(image)
            for i in range(B):
                from medAI.utils.data.patch_extraction import PatchView

                im = image[i].permute(1, 2, 0).cpu().numpy()
                mask = needle_mask[i].permute(1, 2, 0).cpu().numpy()
                pv = PatchView.from_sliding_window(
                    im,
                    window_size=(128, 128),
                    stride=(64, 64),
                    masks=[mask],
                    thresholds=[0.2],
                )
                for position, patch in zip(pv.positions, pv):
                    patches.append(torch.from_numpy(patch).permute(2, 0, 1))
                    positions.append(torch.from_numpy(position))
                    batch_indices.append(i)

            patches = torch.stack(patches).to(DEVICE)
            positions = torch.stack(positions).to(DEVICE)
            positions = positions[:, [1, 0]]
            batch_indices = torch.tensor(batch_indices)
            patch_cnn_output = self.patch_feature_cnn(patches)
            patch_cnn_output = self.patch_feature_prompt_module(patch_cnn_output)
            position_encoding_outputs = (
                self.medsam_model.prompt_encoder.pe_layer.forward_with_coords(
                    positions[None, ...], image_size=(1024, 1024)
                )[0]
            )
            patch_cnn_output = patch_cnn_output + position_encoding_outputs

            sparse_embeddings_by_batch = []
            for i in range(B):
                patch_embeddings_for_batch = patch_cnn_output[batch_indices == i]
                sparse_embeddings_by_batch.append(patch_embeddings_for_batch)

            max_len = max([len(e) for e in sparse_embeddings_by_batch])
            patch_cnn_sparse_embeddings = torch.zeros(B, max_len, 256, device=DEVICE)
            for i, e in enumerate(sparse_embeddings_by_batch):
                patch_cnn_sparse_embeddings[i, : len(e)] = e
                patch_cnn_sparse_embeddings[i, len(e) :] = self.pad_token[None, None, :]

            if self.prompt_dropout > 0 and self.training:
                for i in range(patch_cnn_sparse_embeddings.shape[1]):
                    if torch.rand(1) < self.prompt_dropout:
                        patch_cnn_sparse_embeddings[
                            :, i, :
                        ] = self.null_prompt.repeat_interleave(B, 0)

            sparse_embedding = torch.cat(
                [sparse_embedding, patch_cnn_sparse_embeddings], dim=1
            )

        mask_logits = self.medsam_model.mask_decoder.forward(
            image_feats,
            self.medsam_model.prompt_encoder.get_dense_pe(),
            sparse_embedding,
            dense_embedding,
            multimask_output=False,
        )[0]
        return mask_logits

    def train(self, mode: bool = True):
        super().train(mode)
        if (
            self.sparse_cnn_backbone_path is not None
            and self.patch_feature_cnn is not None
        ):
            self.patch_feature_cnn.eval()

    def get_params_groups(self):
        from itertools import chain

        encoder_parameters = [
            p
            for (k, p) in self.medsam_model.image_encoder.named_parameters()
            if "neck" not in k
        ]
        warmup_parameters = chain(
            self.medsam_model.mask_decoder.parameters(),
            self.task_prompt_module.parameters(),
            self.anatomical_prompt_module.parameters(),
            self.psa_prompt_module.parameters(),
            self.age_prompt_module.parameters(),
            [self.null_prompt],
            [self.data_independent_prompts],
            self.family_history_prompt_module.parameters(),
            self.patch_feature_prompt_module.parameters(),
            self.medsam_model.image_encoder.neck.parameters(),
            self.medsam_model.prompt_encoder.parameters(),
            [self.pad_token],
        )
        cnn_parameters = (
            self.patch_feature_cnn.parameters()
            if self.patch_feature_cnn is not None
            else []
        )

        return encoder_parameters, warmup_parameters, cnn_parameters

    @classmethod
    def add_arguments(cls, parser: ArgumentParser):
        group = parser.add_argument_group("Model")
        group.add_argument(
            "--backbone",
            type=str,
            choices=cls.BACKBONE_OPTIONS,
            default="medsam",
            help="The backbone to use for the model.",
        )
        group.add_argument(
            "--prompts",
            type=str,
            nargs="+",
            default=["task", "anatomical", "psa", "age", "family_history"],
            help="The prompts to use for the model.",
            choices=cls.PROMPT_OPTIONS + ["none"],
        )
        group.add_argument(
            "--prompt_dropout",
            type=float,
            default=0.0,
            help="The dropout to use for the prompts.",
        )
        group.add_argument(
            "--cnn_mode",
            choices=("dense_prompt", "sparse_prompt", "disabled"),
            type=lambda x: None if x == "disabled" else str(x),
            help="Mode to use for the CNN branch.",
            default="disabled",
        )
        group.add_argument(
            "--replace_patch_embed",
            action="store_true",
            help="If True, replaces the patch embedding with a learned convolutional patch embedding.",
        )
        group.add_argument(
            "--sparse_cnn_backbone_path",
            type=str,
            default=None,
            help="The path to the sparse CNN backbone to use. If None, randomly initializes and trains the backbone.",
        )
        group.add_argument(
            "--freeze_mask_decoder",
            action="store_true",
            help="If True, freezes the mask decoder.",
        )
        group.add_argument(
            "--freeze_image_encoder",
            action="store_true",
            help="If True, freezes the image encoder. This is useful for partial finetuning or prompt tuning",
        )
        group.add_argument(
            "--freeze_cnn",
            action="store_true",
            help="If True, freezes the CNN.",
        )
        return group

    @staticmethod
    def from_args(args):
        if "none" in args.prompts:
            args.prompts = []

        print(
            f"""Building model with args: 
    {args.backbone=}
    {args.prompts=}
    {args.prompt_dropout=}
    {args.sparse_cnn_backbone_path=}
    {args.replace_patch_embed=}
"""
        )

        return ProFound(
            sam_backbone=args.backbone,
            prompts=args.prompts,
            prompt_dropout=args.prompt_dropout,
            sparse_cnn_backbone_path=args.sparse_cnn_backbone_path,
            replace_patch_embed=args.replace_patch_embed,
            freeze_mask_decoder=args.freeze_mask_decoder,
            freeze_image_encoder=args.freeze_image_encoder,
            freeze_cnn=args.freeze_cnn,
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
        gce = (1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
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


def build_sam_unetr():
    from medAI.vendor.SAM_UNETR.samunetr.SAMUNETR_V2 import SAMUNETR

    model = SAMUNETR(
        img_size=256,
        in_channels=3,
        out_channels=1,
        trainable_encoder=True,
        pretrained=True,
    )
    return model


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    args = parse_args()
    experiment = Experiment(args)
    experiment.run()
