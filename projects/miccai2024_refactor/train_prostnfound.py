import logging
import os
import typing as tp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from src.data_factory import BModeDataFactoryV1, BModeDataFactoryV1Config
from src.prostnfound import ProstNFound, ProstNFoundConfig, BackboneOptions, PromptOptions
from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy
from pydantic import BaseModel


from medAI.utils.reproducibiliy import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)



class WandbConfig(BaseModel):
    project: str = "miccai2024"
    group: tp.Optional[str] = None
    name: tp.Optional[str] = None
    log_images: bool = False
    tags: tp.List[str] = []


class CancerDetectionValidRegionLossConfig(BaseModel):
    base_loss: str = "ce"
    loss_pos_weight: float = 1.0
    prostate_mask: bool = True
    needle_mask: bool = True


class OptimizerConfig(BaseModel):
    optimizer: str = "adamw"
    cnn_lr: float = 1e-5
    cnn_frozen_epochs: int = 0
    cnn_warmup_epochs: int = 5
    encoder_lr: float = 1e-5
    encoder_frozen_epochs: int = 0
    encoder_warmup_epochs: int = 5
    main_lr: float = 1e-5
    main_frozen_epochs: int = 0
    main_warmup_epochs: int = 5
    wd: float = 0


class Args(BaseModel):
    wandb: WandbConfig = WandbConfig()
    data: BModeDataFactoryV1Config = BModeDataFactoryV1Config(
        test_center="UVA",
        min_involvement_train=40,
        batch_size=1,
        image_size=1024,
        mask_size=256,
        augmentations="translate",
        remove_benign_cores_from_positive_patients=True,
        train_subset_seed=42,
        rf_as_bmode=False,
    )
    model: ProstNFoundConfig = ProstNFoundConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    losses: tp.List[CancerDetectionValidRegionLossConfig] = [
        CancerDetectionValidRegionLossConfig()
    ]
    loss_weights: tp.List[float] = [1.0]

    epochs: int = 30
    cutoff_epoch: tp.Optional[int] = None
    test_every_epoch: bool = False
    accumulate_grad_steps: int = 8
    run_test: bool = False

    # misc 
    encoder_weights_path: tp.Optional[str] = None
    encoder_load_mode: str = "none"
    seed: int = 0 
    use_amp: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir: str = "experiments/default"
    checkpoint_dir: tp.Optional[str] = None
    debug: bool = False
    save_weights: str = "best"
    custom_prompt_table_path: tp.Optional[str] = None   

    def model_post_init(self, __context=None):
        if self.model.sam_backbone in [BackboneOptions.adapter_sammed_2d, BackboneOptions.sam_med2d]: 
            self.data.mask_size = 64 
        else: 
            self.data.mask_size = 256
        if self.data.limit_train_data is None: 
            self.data.limit_train_data = 1.0


class Experiment:
    def __init__(self, config: Args):
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
            self.config.wandb.name = "debug"
        wandb.init(
            project=self.config.wandb.project,
            group=self.config.wandb.group,
            name=self.config.wandb.name,
            config=self.config.model_dump(),
            tags=self.config.wandb.tags,
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

        self.model = ProstNFound(self.config.model)

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
        for i, loss_config in enumerate(self.config.losses):
            loss_name = "valid_region"
            base_loss_name = loss_config.base_loss
            loss_pos_weight = loss_config.loss_pos_weight
            loss_prostate_mask = loss_config.prostate_mask
            loss_needle_mask = loss_config.needle_mask
            loss_weight = self.config.loss_weights[i]

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

        # total_epochs = self.config.epochs
        # encoder_frozen_epochs = self.config.warmup_epochs
        # warmup_epochs = 5
        # niter_per_ep = len(self.train_loader)
        # warmup_lr_factor = self.config.warmup_lr / self.config.lr
        params = [
            {"params": encoder_parameters, "lr": self.config.optimizer.encoder_lr},
            {"params": warmup_parameters, "lr": self.config.optimizer.main_lr},
            {"params": cnn_parameters, "lr": self.config.optimizer.cnn_lr},
        ]

        class LRCalculator:
            def __init__(
                self, frozen_epochs, warmup_epochs, total_epochs, niter_per_ep
            ):
                self.frozen_epochs = frozen_epochs
                self.warmup_epochs = warmup_epochs
                self.total_epochs = total_epochs
                self.niter_per_ep = niter_per_ep

            def __call__(self, iter):
                if iter < self.frozen_epochs * self.niter_per_ep:
                    return 0
                elif (
                    iter < (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                ):
                    return (iter - self.frozen_epochs * self.niter_per_ep) / (
                        self.warmup_epochs * self.niter_per_ep
                    )
                else:
                    cur_iter = (
                        iter
                        - (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                    )
                    total_iter = (
                        self.total_epochs - self.warmup_epochs - self.frozen_epochs
                    ) * self.niter_per_ep
                    return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))

        self.optimizer = AdamW(params, weight_decay=self.config.optimizer.wd)
        from torch.optim.lr_scheduler import LambdaLR
        self.lr_scheduler = LambdaLR(
            self.optimizer,
            [
                LRCalculator(
                    self.config.optimizer.encoder_frozen_epochs,
                    self.config.optimizer.encoder_warmup_epochs,
                    self.config.epochs,
                    len(self.train_loader),
                ),
                LRCalculator(
                    self.config.optimizer.main_frozen_epochs,
                    self.config.optimizer.main_warmup_epochs,
                    self.config.epochs,
                    len(self.train_loader),
                ),
                LRCalculator(
                    self.config.optimizer.cnn_frozen_epochs,
                    self.config.optimizer.cnn_warmup_epochs,
                    self.config.epochs,
                    len(self.train_loader),
                ),
            ],
        )

    def setup_data(self):
        logging.info("Setting up data")

        data_config = deepcopy(self.config.data)
        data_config.include_rf = "sparse_cnn_patch_features_rf" in self.config.model.prompts
        data_config.limit_train_data = (
            data_config.limit_train_data
            if data_config.limit_train_data < 1
            else None
        )

        data_factory = BModeDataFactoryV1(
            data_config
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

        if self.config.custom_prompt_table_path is not None:
            print("Loading custom prompt table")
            self.custom_prompt_table = pd.read_csv(self.config.custom_prompt_table_path)
            self.custom_prompt_table.set_index("core_id", inplace=True, drop=True)
            columns = self.custom_prompt_table.columns
            self.custom_prompt_table_col = columns[0]
            print(f"Using custom prompt {self.custom_prompt_table_col}")
            self.avg_custom_prompt = self.custom_prompt_table[
                self.custom_prompt_table_col
            ].mean()

    def get_custom_prompts(self, core_ids):
        custom_prompts = []
        for core_id in core_ids:
            if core_id not in self.custom_prompt_table.index:
                logging.warning(f"Core id {core_id} not found in custom prompt table")
                custom_prompts.append(self.avg_custom_prompt)
                continue
            custom_prompt = self.custom_prompt_table.loc[
                core_id, self.custom_prompt_table_col
            ].tolist()
            if isinstance(custom_prompt, list):
                custom_prompt = custom_prompt[0]
            custom_prompts.append(custom_prompt)
        return torch.tensor(
            custom_prompts, dtype=torch.float, device=self.config.device
        ).unsqueeze(1)

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
            approx_psa_density = batch["approx_psa_density"].to(self.config.device)

            core_ids = batch["core_id"]
            if self.config.custom_prompt_table_path is not None:
                custom_prompts = self.get_custom_prompts(core_ids)
            else:
                custom_prompts = None

            if "rf" in batch:
                rf = batch.pop("rf").to(self.config.device)
            else:
                rf = None

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
                    approx_psa_density=approx_psa_density,
                    rf_image=rf,
                    custom=custom_prompts,
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
            if train_iter % 100 == 0 and self.config.wandb.log_images:
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
            approx_psa_density = batch["approx_psa_density"].to(self.config.device)

            core_ids = batch["core_id"]
            if self.config.custom_prompt_table_path is not None:
                custom_prompts = self.get_custom_prompts(core_ids)
            else:
                custom_prompts = None

            if "rf" in batch:
                rf = batch.pop("rf").to(self.config.device)
            else:
                rf = None

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
                    approx_psa_density=approx_psa_density,
                    rf_image=rf,
                    custom=custom_prompts,
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

            if train_iter % 100 == 0 and self.config.wandb.log_images:
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
            predictions, labels, log_images=self.config.wandb.log_images
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
                core_probs, core_labels, log_images=self.config.wandb.log_images
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
            predictions, labels, log_images=self.config.wandb.log_images
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
        if self.config.wandb.log_images is False:
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
        approx_psa_density = batch["approx_psa_density"].to(self.config.device)
        if "rf" in batch:
            rf = batch.pop("rf").to(self.config.device)
        else:
            rf = None

        core_ids = batch["core_id"]
        if self.config.custom_prompt_table_path is not None:
            custom_prompts = self.get_custom_prompts(core_ids)
        else:
            custom_prompts = None

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
            approx_psa_density=approx_psa_density,
            custom=custom_prompts,
            rf_image=rf,
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

        if self.config.save_weights == "best":
            if not is_best_score:
                return
            else:
                fname = "best_model.ckpt"
        else:
            fname = f"model_epoch{self.epoch}_auc{score:.2f}.ckpt"

        logging.info("Saving model to checkpoint directory")
        logging.info(f"Checkpoint directory: {self.config.checkpoint_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.checkpoint_dir, fname),
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

            json.dump(dataclass_to_dict(self.config), f)

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
    args = Args(
        wandb=WandbConfig(
            project='miccai2024_repro'
        )
    )
    experiment = Experiment(args)
    experiment.run()
