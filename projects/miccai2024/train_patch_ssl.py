import logging
import os
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from src.patch_model_factory import resnet10t_instance_norm
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from medAI.datasets.nct2013 import data_accessor
from medAI.datasets.nct2013.cohort_selection import (
    apply_core_filters,
    get_core_ids,
    get_patient_splits_by_center,
    select_cohort,
)
from medAI.datasets.nct2013.utils import load_or_create_resized_bmode_data
from medAI.modeling.simclr import SimCLR
from medAI.modeling.vicreg import VICReg
from medAI.utils.data.patch_extraction import PatchView
from medAI.utils.reproducibiliy import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    # fmt: off
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group("Data")
    group.add_argument("--test_center", type=str, default="UVA")
    group.add_argument("--val_seed", type=int, default=0)
    group.add_argument("--data_type", type=str, default="bmode")
    args, _ = parser.parse_known_args()
    if args.data_type == "bmode":
        group.add_argument("--patch_size", type=int, default=128)
        group.add_argument("--stride", type=int, default=32)
    else: 
        group.add_argument("--patch_size_mm", type=float, nargs=2, default=[5.0, 5.0])
        group.add_argument("--patch_stride_mm", type=float, nargs=2, default=[1.0, 1.0])
    group.add_argument("--batch_size", type=int, default=128)
    group.add_argument("--full_prostate", action="store_true", default=False, 
                       help="""Whether to use the full prostate for SSL patches. If False, only select patches within
                       the needle mask.""")
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_weights_path", type=str, default="best_model.pth", help="Path to save the best model weights")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to save and load experiment state")
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    parser.add_argument("--name", type=str, default=None, help="Name of the experiment")

    # fmt: on
    return parser.parse_args()


def main(args):
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        state = torch.load(args.checkpoint_path)
    else:
        state = None
    if os.path.exists(args.save_weights_path) and state is None: 
        print(f"Model weights already exist at {args.save_weights_path}. Exiting.")
        return

    wandb_run_id = state["wandb_run_id"] if state is not None else None
    run = wandb.init(
        project="miccai2024_ssl_debug", config=args, id=wandb_run_id, resume="allow", 
        name=args.name
    )
    wandb_run_id = run.id

    set_global_seed(args.seed)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    ssl_loader, train_loader, val_loader, test_loader = make_data_loaders(args)

    backbone = resnet10t_instance_norm()
    model = VICReg(backbone, proj_dims=[512, 512, 2048], features_dim=512).to(DEVICE)
    if state is not None:
        model.load_state_dict(state["model"])

    from medAI.utils.cosine_scheduler import cosine_scheduler

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])

    cosine_scheduler = cosine_scheduler(
        1e-4, 0, epochs=args.epochs, niter_per_ep=len(ssl_loader)
    )

    best_score = 0.0 if state is None else state["best_score"]
    start_epoch = 0 if state is None else state["epoch"]
    early_stopping_counter = 0 if state is None else state['early_stopping_counter']

    if state is not None:
        set_all_rng_states(state["rng_states"])

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch}")

        if args.checkpoint_path is not None:
            print("Saving checkpoint")
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
                "epoch": epoch,
                "rng_states": get_all_rng_states(),
                "wandb_run_id": wandb_run_id,
                "early_stopping_counter": early_stopping_counter,
            }
            torch.save(state, args.checkpoint_path)

        print("Running SSL")
        model.train()
        for i, batch in enumerate(tqdm(ssl_loader)):
            # set lr
            iter = epoch * len(ssl_loader) + i
            lr = cosine_scheduler[iter]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            wandb.log({"lr": lr})

            optimizer.zero_grad()
            p1, p2 = batch
            p1, p2 = p1.to(DEVICE), p2.to(DEVICE)
            loss = model(p1, p2)
            wandb.log({"ssl_loss": loss.item()})
            loss.backward()
            optimizer.step()

        metrics = run_linear_probing(model, train_loader, val_loader)
        score = metrics["auc"]
        metrics = {f"val/{k}": v for k, v in metrics.items()}
        metrics['epoch'] = epoch

        wandb.log(metrics)

        if score > best_score:
            early_stopping_counter = 0
            best_score = score
            best_model_state = model.state_dict()
            torch.save(best_model_state, args.save_weights_path)
        else: 
            early_stopping_counter += 1
            if early_stopping_counter > 5: 
                print(f"Early stopping after {epoch} epochs with no improvement")
                break


def run_linear_probing(model, train_loader, test_loader):
    print("Running linear probing")
    model.eval()
    from medAI.utils.accumulators import DataFrameCollector

    accumulator = DataFrameCollector()

    X_train = []
    y_train = []
    for i, batch in enumerate(tqdm(train_loader)):
        patch = batch["patch"].to(DEVICE)
        y = torch.tensor(
            [0 if grade == "Benign" else 1 for grade in batch["grade"]],
            dtype=torch.long,
        )
        logging.debug(f"{patch.shape=}, {y.shape=}")
        with torch.no_grad():
            features = model.backbone(patch)
        logging.debug(f"{features.shape=}")
        X_train.append(features)
        y_train.append(y)
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train)

    X_val = []
    for i, batch in enumerate(tqdm(test_loader)):
        patch = batch.pop("patch").to(DEVICE)
        accumulator(batch)
        with torch.no_grad():
            features = model.backbone(patch)
        X_val.append(features)

    X_val = torch.cat(X_val, dim=0)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    y_pred = clf.predict_proba(X_val.cpu().numpy())
    table = accumulator.compute()
    accumulator.reset()
    # insert predictions into table
    table.loc[:, "y_pred"] = y_pred[:, 1]

    y_pred_core = table.groupby("core_id")["y_pred"].mean()
    table["label"] = table.grade.apply(lambda g: 0 if g == "Benign" else 1)
    y_true_core = table.groupby("core_id")["label"].first()
    score = roc_auc_score(y_true_core, y_pred_core)

    high_involvement_table = table[(table.pct_cancer > 40) | (table.grade == "Benign")]
    y_true_high_involvement = high_involvement_table.groupby("core_id")["label"].first()
    y_pred_high_involvement = high_involvement_table.groupby("core_id")["y_pred"].mean()
    score_high_involvement = roc_auc_score(
        y_true_high_involvement, y_pred_high_involvement
    )

    return {
        "auc": score,
        "auc_high_involvement": score_high_involvement,
    }


def make_data_loaders(args):
    print(f"Preparing data loaders for test center {args.test_center}")
    from src.dataset import BModePatchesDataset, RFPatchesDataset, Transform, SSLTransform

    train_patients, val_patients, test_patients = get_patient_splits_by_center(
        args.test_center, val_size=0.2, val_seed=args.val_seed
    )

    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")

    ssl_train_core_ids = get_core_ids(train_patients)
    train_core_ids = apply_core_filters(
        ssl_train_core_ids.copy(),
        exclude_benign_cores_from_positive_patients=True,
        undersample_benign_ratio=1,
        involvement_threshold_pct=40,
    )
    val_core_ids = get_core_ids(val_patients)
    test_core_ids = get_core_ids(test_patients)

    print(f"SSL Train cores: {len(ssl_train_core_ids)}")
    print(f"Train cores: {len(train_core_ids)}")
    print(f"Val cores: {len(val_core_ids)}")
    print(f"Test cores: {len(test_core_ids)}")

    if args.data_type == "bmode": 
        print("SSL dataset...")
        ssl_dataset = BModePatchesDataset(
            ssl_train_core_ids,
            patch_size=(args.patch_size, args.patch_size),
            stride=(args.stride, args.stride),
            needle_mask_threshold=0.6 if not args.full_prostate else -1,
            prostate_mask_threshold=-1 if not args.full_prostate else 0.1,
            transform=SSLTransform(),
        )
        print("Train dataset...")
        train_dataset = BModePatchesDataset(
            train_core_ids,
            patch_size=(args.patch_size, args.patch_size),
            stride=(args.stride, args.stride),
            needle_mask_threshold=0.6,
            prostate_mask_threshold=-1,
            transform=Transform(),
        )
        print("Val dataset...")
        val_dataset = BModePatchesDataset(
            val_core_ids,
            patch_size=(args.patch_size, args.patch_size),
            stride=(args.stride, args.stride),
            needle_mask_threshold=0.6,
            prostate_mask_threshold=-1,
            transform=Transform(),
        )
        print("Test dataset...")
        test_dataset = BModePatchesDataset(
            test_core_ids,
            patch_size=(args.patch_size, args.patch_size),
            stride=(args.stride, args.stride),
            needle_mask_threshold=0.6,
            prostate_mask_threshold=-1,
            transform=Transform(),
        )
    else: 
        print("SSL dataset...")
        ssl_dataset = RFPatchesDataset(
            ssl_train_core_ids,
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=0.1,
            transform=SSLTransform(),
        )
        print("Train dataset...")
        train_dataset = RFPatchesDataset(
            train_core_ids,
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=0.1,
            transform=Transform(),
        )
        print("Val dataset...")
        val_dataset = RFPatchesDataset(
            val_core_ids,
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=0.1,
            transform=Transform(),
        )
        print("Test dataset...")
        test_dataset = RFPatchesDataset(
            test_core_ids,
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=0.1,
            transform=Transform(),
        )

    ssl_loader = torch.utils.data.DataLoader(
        ssl_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"SSL Train batches: {len(ssl_loader)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return ssl_loader, train_loader, val_loader, test_loader



if __name__ == "__main__":
    args = parse_args()
    main(args)
