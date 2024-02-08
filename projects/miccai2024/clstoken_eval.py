import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import wandb
from src.data_factory import BModeDataFactoryV1
from train_medsam_dino_style import MedSAMDino

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--test_center", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--augmentations", type=str, default="none")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cosine_scheduler", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--eval_mode",
        choices=["linear", "full_finetune", "partial_finetune"],
        default="linear",
        help="linear: only linear layer is trained, full_finetune: all layers are trained, partial_finetune: the class-token related attention blocks are trained",
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true", default=False)
    return parser.parse_args()


def main(args):
    wandb.init(project="debug", config=args)

    df = BModeDataFactoryV1(
        fold=args.fold,
        n_folds=args.n_folds,
        test_center=args.test_center,
        batch_size=args.batch_size,
        augmentations=args.augmentations,
    )
    train_loader = df.train_loader()
    test_loader = df.test_loader()

    model = MedSAMDino()
    model.load_state_dict(torch.load(args.weights_path, map_location=DEVICE))

    if args.eval_mode == "linear":
        for param in model.parameters():
            param.requires_grad = False
    elif args.eval_mode == "full_finetune":
        ...
    elif args.eval_mode == "partial_finetune":
        for param in model.parameters():
            param.requires_grad = False
        for (
            param
        ) in model.image_encoder_wrapped.class_token_to_image_attns.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown eval_mode: {args.eval_mode}")

    lin_layer = nn.Linear(2024, 1)  # predict cancer or not
    model = nn.Sequential(model, nn.Flatten(), lin_layer)
    model.to(DEVICE)
    torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    if args.cosine_scheduler:
        from medAI.utils.cosine_scheduler import cosine_scheduler

        lr_schedule = cosine_scheduler(
            args.lr, 1e-6, args.epochs, len(train_loader), warmup_epochs=5
        )
    else:
        lr_schedule = None

    for epoch in range(args.epochs):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            scaler=scaler,
            lr_schedule=lr_schedule,
            accumulate_grad_steps=args.accumulate_grad_batches,
        )
        eval_one_epoch(model, test_loader, epoch, desc="test")


def train_one_epoch(
    model,
    loader,
    optimizer,
    epoch,
    lr_schedule=None,
    scaler=None,
    device=DEVICE,
    accumulate_grad_steps=1,
):
    model.train()

    for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch} training")):
        # Compute loss
        bmode = batch["bmode"].to(device)
        # wrap label for BCE loss
        label = batch["label"].to(device).unsqueeze(-1).float()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(bmode)
            loss = nn.functional.binary_cross_entropy_with_logits(pred, label)

            # Optimize
            if lr_schedule is not None:
                current_lr = lr_schedule[i + len(loader) * epoch]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accumulate_grad_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

        metrics = {"train_loss": loss.item()}
        metrics["lr"] = optimizer.param_groups[0]["lr"]

        wandb.log(metrics)


@torch.no_grad()
def eval_one_epoch(model, loader, epoch, device=DEVICE, desc="val"):
    model.eval()

    pred = []
    labels = []
    involvement = []

    for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch} evaluation")):
        bmode = batch["bmode"].to(device)
        label = batch["label"].to(device)
        involvement_ = batch["involvement"].to(device)
        pred_ = model(bmode)

        pred.append(pred_)
        labels.append(label)
        involvement.append(involvement_)

    pred = torch.cat(pred)
    labels = torch.cat(labels)
    involvement = torch.cat(involvement)

    return create_and_report_metrics(
        pred,
        labels,
        involvement,
        epoch,
        desc=desc,
    )


def create_and_report_metrics(
    mean_predictions_in_mask,
    labels,
    involvement,
    epoch,
    desc="eval",
):
    """
    Calculate and report metrics for the given predictions and labels.
    """

    mean_predictions_in_mask = mean_predictions_in_mask.cpu().numpy()
    labels = labels.cpu().numpy()
    involvement = involvement.cpu().numpy()

    from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score

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
        metrics["core_auc_high_involvement"] = roc_auc_score(core_labels, core_probs)
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

    metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    wandb.log(metrics)
    return metrics


if __name__ == "__main__":
    main(parse_args())
