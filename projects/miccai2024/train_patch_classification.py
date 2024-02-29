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
    group.add_argument("--batch_size", type=int, default=128)
    group.add_argument("--undersample_benign_ratio_train", type=float, default=None)

    args, _ = parser.parse_known_args()
    type = args.data_type
    if type == "bmode":
        group.add_argument("--patch_size", type=int, default=128)
        group.add_argument("--stride", type=int, default=32)
    elif type == "rf":
        group.add_argument("--patch_size_mm", type=float, nargs=2, default=[5, 5])
        group.add_argument("--patch_stride_mm", type=float, nargs=2, default=[1, 1])

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--load_weights_path", type=str, default=None, help="Path to load the backbone weights")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to save and load experiment state")

    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)

    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit")
    # fmt: on
    return parser.parse_args()


def main(args):
    set_global_seed(args.seed)

    wandb.init(
        project="miccai2024_patch_classification",
        config=args,
        name=args.name,
        group=args.group,
    )

    train_loader, val_loader, test_loader = make_data_loaders(args)

    model = resnet10t_instance_norm()
    if args.load_weights_path:
        weights = torch.load(args.load_weights_path)
        weights = {
            k.replace("backbone.", ""): v for k, v in weights.items() if "backbone" in k
        }
        model.load_state_dict(weights)
    model = torch.nn.Sequential(model, torch.nn.Linear(512, 2))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    model.train()
    from medAI.utils.accumulators import DataFrameCollector

    accumulator = DataFrameCollector()

    for epoch in range(args.epochs):
        accumulator.reset()
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            patch = batch.pop("patch").to(DEVICE)
            y = torch.tensor(
                [0 if grade == "Benign" else 1 for grade in batch["grade"]],
                dtype=torch.long,
            ).to(DEVICE)
            prediction = model(patch)
            loss = criterion(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item()})

            accumulator({"pred": prediction.softmax(-1), "y": y, **batch})

        train_outputs = accumulator.compute()
        wandb.log(
            {f"train_{k}": v for k, v in calculate_metrics(train_outputs).items()}
        )
        accumulator.reset()
        model.eval()

        for i, batch in enumerate(tqdm(val_loader)):
            patch = batch.pop("patch").to(DEVICE)
            y = torch.tensor(
                [0 if grade == "Benign" else 1 for grade in batch["grade"]],
                dtype=torch.long,
            ).to(DEVICE)
            prediction = model(patch)
            accumulator({"pred": prediction.softmax(-1), "y": y, **batch})

        val_outputs = accumulator.compute()
        wandb.log({f"val_{k}": v for k, v in calculate_metrics(val_outputs).items()})
        accumulator.reset()
        model.eval()

        for i, batch in enumerate(tqdm(test_loader)):
            patch = batch.pop("patch").to(DEVICE)
            y = torch.tensor(
                [0 if grade == "Benign" else 1 for grade in batch["grade"]],
                dtype=torch.long,
            ).to(DEVICE)
            prediction = model(patch)
            accumulator({"pred": prediction.softmax(-1), "y": y, **batch})

        test_outputs = accumulator.compute()
        wandb.log({f"test_{k}": v for k, v in calculate_metrics(test_outputs).items()})

        scheduler.step()


def calculate_metrics(results_table):
    from src.utils import calculate_metrics

    core_pred = results_table.groupby("core_id").pred_1.mean()
    core_y = results_table.groupby("core_id").y.first()
    core_involvement = results_table.groupby("core_id").pct_cancer.first()

    metrics = {}
    metrics.update(calculate_metrics(core_pred, core_y))

    core_pred_high_involvement = core_pred[(core_involvement > 40) | (core_y == 0)]
    core_y_high_involvement = core_y[(core_involvement > 40) | (core_y == 0)]

    metrics.update(
        {
            f"{k}_high_inv": v
            for k, v in calculate_metrics(
                core_pred_high_involvement, core_y_high_involvement
            ).items()
        }
    )

    return metrics


def make_data_loaders(args):
    from src.dataset import BModePatchesDataset, RFPatchesDataset, Transform

    print(f"Preparing data loaders for test center {args.test_center}")

    train_patients, val_patients, test_patients = get_patient_splits_by_center(
        args.test_center, val_size=0.2, val_seed=args.val_seed
    )

    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")

    train_core_ids = get_core_ids(train_patients)
    train_core_ids = apply_core_filters(
        train_core_ids,
        exclude_benign_cores_from_positive_patients=True,
        undersample_benign_ratio=args.undersample_benign_ratio_train,
        involvement_threshold_pct=40,
    )
    val_core_ids = get_core_ids(val_patients)
    test_core_ids = get_core_ids(test_patients)

    print(f"Train cores: {len(train_core_ids)}")
    print(f"Val cores: {len(val_core_ids)}")
    print(f"Test cores: {len(test_core_ids)}")

    if args.data_type == "bmode":
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
        print("Train dataset...")
        train_dataset = RFPatchesDataset(
            train_core_ids,
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=-1,
            transform=Transform(),
        )
        print("Val dataset...")
        val_dataset = RFPatchesDataset(
            val_core_ids,
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=-1,
            transform=Transform(),
        )
        print("Test dataset...")
        test_dataset = RFPatchesDataset(
            test_core_ids,
            patch_size_mm=args.patch_size_mm,
            patch_stride_mm=args.patch_stride_mm,
            needle_mask_threshold=0.6,
            prostate_mask_threshold=-1,
            transform=Transform(),
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
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    args = parse_args()
    main(args)
