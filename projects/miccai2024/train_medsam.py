import hydra 
import torch 
import torch.nn as nn
import os 
from dataclasses import dataclass
from src.data_factory import BModeDataFactoryV1
from matplotlib import pyplot as plt
import medAI
from torch import optim
import logging
from tqdm import tqdm
import wandb
from abc import ABC, abstractmethod
from einops import rearrange, repeat
from src.masked_prediction_module import MaskedPredictionModule
from medAI.modeling.sam import MedSAMForFinetuning
from omegaconf import OmegaConf
from medAI.utils.reproducibiliy import set_global_seed, get_all_rng_states, set_all_rng_states



@hydra.main(config_path="conf", config_name="train_medsam")
def main(cfg):
    Experiment(cfg)()


class Experiment:
    def __init__(self, config):
        logging.info("Setting up experiment")
        logging.info("Running in directory: " + os.getcwd())
        self.config = config
        wandb.init(
            **config.wandb,
            config=OmegaConf.to_container(config, resolve=True),
            dir=hydra.utils.get_original_cwd(),
        )
        logging.info("Wandb initialized")
        logging.info("Wandb url: " + wandb.run.url)

        if 'experiment_state.pth' in os.listdir('.'):
            logging.info("Loading experiment state from experiment_state.pth")
            self.state = torch.load('experiment_state.pth')
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

        match self.config.optimizer:
            case "adam":
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.config.lr,
                    weight_decay=self.config.wd,
                )
            case "sgdw":
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.config.lr,
                    momentum=0.9,
                    weight_decay=self.config.wd,
                )
        self.scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=5 * len(self.train_loader),
            max_epochs=self.config.epochs * len(self.train_loader),
        )

        if self.state is not None:
            self.optimizer.load_state_dict(self.state["optimizer"])
            self.scheduler.load_state_dict(self.state["scheduler"])

        self.masked_prediction_module_train = MaskedPredictionModule(
            needle_mask_threshold=self.config.needle_threshold,
            prostate_mask_threshold=self.config.prostate_threshold,
        )
        self.masked_prediction_module_test = MaskedPredictionModule()
        self.gradient_scaler = torch.cuda.amp.GradScaler()

        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        logging.info(
            f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )
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
        self.model.to(self.config.device)
        torch.compile(self.model)

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
        
    def __call__(self):
        for self.epoch in range(self.epoch, self.config.epochs):
            logging.info(f"Epoch {self.epoch}")
            self.save()
            self.training = True
            self.run_epoch(self.train_loader, desc="train")
            self.training = False
            val_metrics = self.run_epoch(self.val_loader, desc="val")
            tracked_metric = val_metrics["val/core_auc_high_involvement"]
            if tracked_metric > self.best_score:
                self.best_score = tracked_metric
                logging.info(f"New best score: {self.best_score}")
                metrics = self.run_epoch(self.test_loader, desc="test")
                test_score = metrics["test/core_auc_high_involvement"]
                if self.config.get('checkpoint_dir', None):
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.checkpoint_dir,
                            f"best_model_epoch{self.epoch}_auc{test_score:.2f}.ckpt",
                        ),
                    )

    def run_epoch(self, loader, desc="train"):
        train = self.training

        with torch.no_grad() if not train else torch.enable_grad():
            self.model.train() if train else self.model.eval()

            core_probs = []
            core_labels = []
            patch_probs = []
            patch_labels = []
            pred_involvement = []
            gt_involvement = []
            patch_involvement = []

            acc_steps = 1
            for i, batch in enumerate(tqdm(loader, desc=desc)):
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
                    masked_prediction_module: MaskedPredictionModule = (
                        self.masked_prediction_module_train
                        if train
                        else self.masked_prediction_module_test
                    )
                    outputs: MaskedPredictionModule.Output = masked_prediction_module(
                        heatmap_logits, needle_mask, prostate_mask, label
                    )

                    loss = self.compute_loss(outputs, involvement)

                if train:
                    self.gradient_scaler.scale(loss).backward()
                    
                    if acc_steps % self.config.accumulate_grad_steps == 0:
                        self.gradient_scaler.step(self.optimizer)
                        self.gradient_scaler.update()
                        self.optimizer.zero_grad()
                        acc_steps = 1
                    else:
                        acc_steps += 1
                    self.scheduler.step()
                    wandb.log(
                        {"train_loss": loss, "lr": self.scheduler.get_last_lr()[0]}
                    )

                core_probs.append(outputs.core_predictions.detach().cpu())
                core_labels.append(outputs.core_labels.detach().cpu())
                patch_probs.append(outputs.patch_predictions.detach().cpu())
                patch_labels.append(outputs.patch_labels.detach().cpu())
                patch_predictions = (outputs.patch_predictions > 0.5).float()
                pred_involvement_batch = []
                for core_idx in outputs.core_indices.unique():
                    pred_involvement_batch.append(
                        patch_predictions[outputs.core_indices == core_idx].mean()
                    )
                pred_involvement.append(
                    torch.stack(pred_involvement_batch).detach().cpu()
                )
                patch_involvement_i = torch.zeros_like(patch_predictions)
                for core_idx in range(len(involvement)):
                    patch_involvement_i[outputs.core_indices == core_idx] = involvement[
                        core_idx
                    ]
                patch_involvement.append(patch_involvement_i.detach().cpu())

                valid_involvement_for_batch = torch.stack([involvement[i] for i in outputs.core_indices.unique()])
                gt_involvement.append(valid_involvement_for_batch.detach().cpu())

                interval = 100 if train else 10
                if i % interval == 0:
                    self.show_example(batch)
                    wandb.log({f"{desc}_example": wandb.Image(plt)})
                    plt.close()

            from sklearn.metrics import roc_auc_score, balanced_accuracy_score, r2_score

            metrics = {}

            # core predictions
            core_probs = torch.cat(core_probs)
            core_labels = torch.cat(core_labels)
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
            pred_involvement = torch.cat(pred_involvement)
            gt_involvement = torch.cat(gt_involvement)
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

    def compute_loss(self, outputs: MaskedPredictionModule.Output, involvement):
        BATCH_SIZE = outputs.core_predictions.shape[0]

        match self.config.loss:
            case "basic_ce":
                loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs.patch_logits, outputs.patch_labels
                )
            case "involvement_tolerant_loss":
                loss = torch.tensor(
                    0, dtype=torch.float32, device=outputs.core_predictions.device
                )
                for i in range(BATCH_SIZE):
                    patch_logits_for_core = outputs.patch_logits[
                        outputs.core_indices == i
                    ]
                    patch_labels_for_core = outputs.patch_labels[
                        outputs.core_indices == i
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

            case _:
                raise ValueError(f"Unknown loss: {self.config.loss}")
            
        return loss

    @torch.no_grad()
    def show_example(self, batch):
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
        masked_prediction_module = (
            self.masked_prediction_module_train
            if self.training
            else self.masked_prediction_module_test
        )
        outputs = masked_prediction_module(logits, needle_mask, prostate_mask, label)
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
        logging.info("Saving experiment snapshot")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
                "rng": get_all_rng_states(),
            },
            "experiment.ckpt",
        )

    def checkpoint(self):
        self.save()
        return super().checkpoint()

    
if __name__ == "__main__":
    main()