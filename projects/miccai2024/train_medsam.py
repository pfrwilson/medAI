import hydra 
import torch 
import torch.nn as nn
import os 
from dataclasses import dataclass
from src.data_factory import BModeDataFactoryV1, AlignedFilesSegmentationDataFactory
from matplotlib import pyplot as plt
import medAI
from torch import optim
import logging
from tqdm import tqdm
import wandb
import numpy as np
from abc import ABC, abstractmethod
from einops import rearrange, repeat
# from src.masked_prediction_module import MaskedPredictionModule
from medAI.modeling.sam import MedSAMForFinetuning
from omegaconf import OmegaConf
from medAI.utils.reproducibiliy import set_global_seed, get_all_rng_states, set_all_rng_states


@hydra.main(config_path="conf", config_name="train_medsam")
def main(cfg):
    exp = Experiment(cfg)
    exp.run()


def involvement_tolerant_loss(patch_logits, patch_labels, core_indices, involvement):
    batch_size = len(involvement    )
    loss = torch.tensor(
        0, dtype=torch.float32, device=patch_logits.device
    )
    for i in range(batch_size):
        patch_logits_for_core = patch_logits[
            core_indices == i
        ]
        patch_labels_for_core = patch_labels[
            core_indices == i
        ]
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
            patch_predictions_for_core_for_loss = (
                patch_logits_for_core[
                    : int(
                        n_predictions * involvement_for_core.item()
                    )
                ]
            )
            patch_labels_for_core_for_loss = patch_labels_for_core[
                : int(n_predictions * involvement_for_core.item())
            ]
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_predictions_for_core_for_loss,
                patch_labels_for_core_for_loss,
            )


class Experiment:
    def __init__(self, config):
        self.config = config

    def setup(self):
        logging.info("Setting up experiment")
        logging.info("Running in directory: " + os.getcwd())

        wandb.init(
            **self.config.wandb,
            config=OmegaConf.to_container(self.config, resolve=True),
            dir=hydra.utils.get_original_cwd(),
        )
        logging.info("Wandb initialized")
        logging.info("Wandb url: " + wandb.run.url)
        # for reproducibility, save full hydra config
        wandb.save(".hydra/**")

        self.exp_state_path = self.config.get('exp_state_path', None) or \
            os.path.join(os.getcwd(), 'experiment_state.pth')
        if os.path.exists(self.exp_state_path):
            logging.info("Loading experiment state from experiment_state.pth")
            self.state = torch.load(self.exp_state_path)
        else:
            logging.info("No experiment state found - starting from scratch") 
            self.state = None 

        set_global_seed(self.config.get('seed', 42))

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

        self.epoch = 0 if self.state is None else self.state["epoch"]
        logging.info(f"Starting at epoch {self.epoch}")
        self.best_score = 0 if self.state is None else self.state["best_score"]
        logging.info(f"Best score so far: {self.best_score}")
        if self.state is not None: 
            rng_state = self.state['rng']
            set_all_rng_states(rng_state)

    def setup_model(self):
        logging.info("Setting up model")
        self.model = MedSAMForFinetuning(freeze_backbone=self.config.get('freeze_backbone', False))
        if self.config.encoder_weights_path is not None: 
            self.model.medsam_model.image_encoder.load_state_dict(torch.load(self.config.encoder_weights_path))
        self.model.to(self.config.device)
        torch.compile(self.model)

    def setup_optimizer(self):
        match self.config.optimizer:
            case "adam":
                opt_factory = lambda parameters, lr: optim.Adam(
                    parameters,
                    lr=lr,
                    weight_decay=self.config.wd,
                )
            case "sgdw":
                opt_factory = lambda parameters, lr: optim.SGD(
                    parameters,
                    lr=lr,
                    momentum=0.9,
                    weight_decay=self.config.wd,
                )

        params = [
            {'params': self.model.medsam_model.image_encoder.parameters(), 'lr': self.config.encoder_lr},
            {'params': self.model.medsam_model.mask_decoder.parameters(), 'lr': self.config.lr},
        ]
        self.optimizer = opt_factory(params, lr=self.config.lr)

        self.scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=5 * len(self.train_loader),
            max_epochs=self.config.epochs * len(self.train_loader),
        )

        self.warmup_optimizer = opt_factory(
            self.model.medsam_model.mask_decoder.parameters(), lr = self.config.warmup_lr
        )

    def setup_data(self):
        logging.info("Setting up data")
        data_factory = BModeDataFactoryV1(
            **self.config.data
        )
        self.train_loader = data_factory.train_loader()
        self.val_loader = data_factory.val_loader()
        self.test_loader = data_factory.test_loader()
        logging.info(f'Number of training batches: {len(self.train_loader)}')
        logging.info(f'Number of validation batches: {len(self.val_loader)}')
        logging.info(f'Number of test batches: {len(self.test_loader)}')
        logging.info(f'Number of training samples: {len(self.train_loader.dataset)}')
        logging.info(f'Number of validation samples: {len(self.val_loader.dataset)}')
        logging.info(f'Number of test samples: {len(self.test_loader.dataset)}')

        # dump core_ids to file 
        train_core_ids = self.train_loader.dataset.core_ids
        val_core_ids = self.val_loader.dataset.core_ids
        test_core_ids = self.test_loader.dataset.core_ids

        with open('train_core_ids.txt', 'w') as f:
            f.write('\n'.join(train_core_ids))
        with open('val_core_ids.txt', 'w') as f:
            f.write('\n'.join(val_core_ids))
        with open('test_core_ids.txt', 'w') as f:
            f.write('\n'.join(test_core_ids))

        wandb.save('train_core_ids.txt')
        wandb.save('val_core_ids.txt')
        wandb.save('test_core_ids.txt')

        if self.config.get('segmentation_data', None) is not None:
            data_factory_segmentation = AlignedFilesSegmentationDataFactory(
                **self.config.segmentation_data
            )
            self.train_loader_segmentation = data_factory_segmentation.train_loader()
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
                
            if new_record or self.config.get('test_every_epoch', False):
                self.training = False
                logging.info("Running test set")
                metrics = self.run_eval_epoch(self.test_loader, desc="test")
                test_score = metrics["test/core_auc_high_involvement"]
            else: 
                test_score = None  

            self.save_model_weights(score=test_score, is_best_score=new_record)            

        logging.info("Finished training")
        self.teardown()

    def train_step(self, batch): 
        bmode = batch["bmode"]
        needle_mask = batch["needle_mask"]
        prostate_mask = batch["prostate_mask"]
        label = batch["label"]
        involvement = batch["involvement"]

        bmode = bmode.to(self.config.device)
        needle_mask = needle_mask.to(self.config.device)
        prostate_mask = prostate_mask.to(self.config.device)
        label = label.to(self.config.device)

        with torch.cuda.amp.autocast(): 
            heatmap_logits = self.model(bmode)

            masks = [] 
            for i in range(len(heatmap_logits)): 
                mask = prostate_mask[i] > self.config.prostate_threshold
                mask = mask & (needle_mask[i] > self.config.needle_threshold)
                masks.append(mask)
            masks = torch.stack(masks)

            predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)
            labels = torch.zeros(len(predictions), device=predictions.device)
            for i in range(len(predictions)): 
                labels[i] = label[batch_idx[i]]
            labels = labels[..., None] # needs to match N, C shape of preds

            if self.config.loss == 'basic_ce':
                loss = nn.functional.binary_cross_entropy_with_logits(
                    predictions, labels
                )

            return loss 

    @torch.no_grad()
    def eval_step(self, batch, aggregate=True):
        bmode = batch["bmode"]
        needle_mask = batch["needle_mask"]
        prostate_mask = batch["prostate_mask"]
        label = batch["label"]
        involvement = batch["involvement"]
        B = len(bmode)

        bmode = bmode.to(self.config.device)
        needle_mask = needle_mask.to(self.config.device)
        prostate_mask = prostate_mask.to(self.config.device)
        label = label.to(self.config.device)

        with torch.cuda.amp.autocast(): 
            heatmap_logits = self.model(bmode)
            
            masks = [] 
            for i in range(len(heatmap_logits)): 
                mask = prostate_mask[i] > 0.5
                mask = mask & (needle_mask[i] > 0.5)
                masks.append(mask)
            masks = torch.stack(masks)

            predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)

            mean_predictions_in_mask = []
            for i in range(B):
                mean_predictions_in_mask.append(predictions[batch_idx == i].sigmoid().mean())
            mean_predictions_in_mask = torch.stack(mean_predictions_in_mask)
            
            if aggregate: 
                self.mean_predictions_in_mask.append(mean_predictions_in_mask)
                self.labels.append(label)
                self.involvement.append(involvement)
        
        return mean_predictions_in_mask, label, involvement

    def run_train_epoch(self, loader, desc="train"):
        self.model.train()
        
        if self.epoch < self.config.warmup_epochs:
            optimizer = self.warmup_optimizer
            scheduler = None
        else:
            optimizer = self.optimizer
            scheduler = self.scheduler

        for i, batch in enumerate(tqdm(loader, desc=desc)):
            if self.config.get('limit_train_batches', None) is not None and i > self.config.limit_train_batches:
                break

            loss = self.train_step(batch)
            self.gradient_scaler.scale(loss).backward()
            
            if i % self.config.accumulate_grad_steps == 0:
                self.gradient_scaler.step(optimizer)
                self.gradient_scaler.update()
                optimizer.zero_grad()
                acc_steps = 1
            else:
                acc_steps += 1

            metrics = {"train_loss": loss.item() / acc_steps}
            metrics['lr'] = optimizer.param_groups[0]['lr']

            if scheduler is not None:
                scheduler.step()
            
            wandb.log(
                metrics
            )

            if i % 100 == 0:
                self.show_example(batch)
                wandb.log({f"{desc}_example": wandb.Image(plt)})
                plt.close()

    def run_eval_epoch(self, loader, desc="eval"):
        self.model.eval()

        self.mean_predictions_in_mask = []
        self.labels = []
        self.involvement = []

        for i, batch in enumerate(tqdm(loader, desc=desc)):
            if self.config.get('limit_val_batches', None) is not None and i > self.config.limit_val_batches:
                break

            self.eval_step(batch)

            if i % 100 == 0:
                self.show_example(batch)
                wandb.log({f"{desc}_example": wandb.Image(plt)})
                plt.close()

        self.mean_predictions_in_mask = torch.cat(self.mean_predictions_in_mask)
        self.labels = torch.cat(self.labels)
        self.involvement = torch.cat(self.involvement)

        return self.create_and_report_metrics(self.mean_predictions_in_mask, self.labels, self.involvement, desc=desc)

    def create_and_report_metrics(self, mean_predictions_in_mask, labels, involvement, desc="eval"):
        mean_predictions_in_mask = mean_predictions_in_mask.cpu().numpy()
        labels = labels.cpu().numpy()
        involvement = involvement.cpu().numpy()

        from sklearn.metrics import roc_auc_score, balanced_accuracy_score, r2_score
        metrics = {}

        # core predictions
        core_probs = mean_predictions_in_mask
        core_labels = labels
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

        # high involvement core predictions
        high_involvement = involvement > 0.4
        benign = core_labels == 0

        keep = np.logical_or(high_involvement, benign)
        if keep.sum() > 0:
            core_probs = core_probs[keep]
            core_labels = core_labels[keep]
            metrics["core_auc_high_involvement"] = roc_auc_score(
                core_labels, core_probs
            )
            plt.hist(
                core_probs[core_labels == 0], bins=100, alpha=0.5, density=True
            )
            plt.hist(
                core_probs[core_labels == 1], bins=100, alpha=0.5, density=True
            )
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

        metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = self.epoch
        wandb.log(metrics)
        return metrics

    @torch.no_grad()
    def show_example(self, batch):
        # don't log images by default, since they take up a lot of space. 
        # should be considered more of a debugging/demonstration tool
        if self.config.get('log_images', False) is False: 
            return

        image = batch["bmode"]
        needle_mask = batch["needle_mask"]
        prostate_mask = batch["prostate_mask"]
        label = batch["label"]
        involvement = batch["involvement"]

        image = image.to(self.config.device)
        needle_mask = needle_mask.to(self.config.device)
        prostate_mask = prostate_mask.to(self.config.device)
        label = label.to(self.config.device)

        logits = self.model(image) 
        mean_predictions_in_mask, label, involvement = self.eval_step(batch, aggregate=False)
        core_prediction = mean_predictions_in_mask[0].item()

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
        
        valid_loss_region = (prostate_mask[0][0] > 0.5).float() * (
            needle_mask[0][0] > 0.5
        ).float()

        alpha = torch.nn.functional.interpolate(
            valid_loss_region[None, None], size=(256, 256), mode="nearest"
        )[0, 0]
        ax[2].imshow(pred[0, 0], alpha=alpha, **kwargs)
        ax[2].set_title(f"Core prediction: {core_prediction:.3f}")

    def save_experiment_state(self):
        if self.config.get('exp_state_path') is None: 
            return
        logging.info(f"Saving experiment snapshot to {self.exp_state_path}")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
                "rng": get_all_rng_states(),
            },
            self.config.exp_state_path,
        )

    def save_model_weights(self, score, is_best_score=False): 
        if self.config.get('checkpoint_dir') is None or not is_best_score: 
            return 
        logging.info("Saving model to checkpoint directory")
        logging.info(f"Checkpoint directory: {self.config.checkpoint_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.checkpoint_dir,
                f"best_model_epoch{self.epoch}_auc{score:.2f}.ckpt",
            ),
        )

    def teardown(self): 
        # remove checkpoint file since it will take up space
        os.remove('experiment_state.ckpt')


class MaskedPredictionModule(nn.Module):
    """
    Computes the patch and core predictions and labels within the valid loss region for a heatmap.
    """
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, heatmap_logits, mask):
        """Computes the patch and core predictions and labels within the valid loss region."""
        B, C, H, W = heatmap_logits.shape

        mask = mask.float()
        mask = torch.nn.functional.interpolate(
            mask, size=(H, W)
        ) > 0.5

        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[..., 0]
        logits_flattened = rearrange(heatmap_logits, "b c h w -> (b h w) c", h=H, w=W)

        logits = logits_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_logits = logits

        return patch_logits, core_idx



if __name__ == "__main__":
    main()