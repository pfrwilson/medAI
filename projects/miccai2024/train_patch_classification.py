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
    parser = ArgumentParser()
    group = parser.add_argument_group("Data")
    group.add_argument("--test_center", type=str, default="UVA")
    group.add_argument("--val_seed", type=int, default=0)
    group.add_argument("--patch_size", type=int, default=128)
    group.add_argument("--stride", type=int, default=32)
    group.add_argument("--batch_size", type=int, default=128)
    group.add_argument("--undersample_benign_ratio_train", type=float, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--load_weights_path", type=str, default=None, help="Path to load the backbone weights")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to save and load experiment state")

    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)

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


class BModePatchesDataset(Dataset):
    _bmode_data, _core_id_2_idx = load_or_create_resized_bmode_data((1024, 1024))
    _metadata_table = data_accessor.get_metadata_table()

    def __init__(
        self,
        core_ids,
        patch_size,
        stride,
        needle_mask_threshold,
        prostate_mask_threshold,
        transform=None,
    ):
        self.core_ids = sorted(core_ids)
        N = len(self.core_ids)

        self._images = [
            self._bmode_data[self._core_id_2_idx[core_id]] for core_id in core_ids
        ]
        self._prostate_masks = np.zeros((N, 256, 256))
        for i, core_id in enumerate(core_ids):
            self._prostate_masks[i] = data_accessor.get_prostate_mask(core_id)
        self._needle_masks = np.zeros((N, 512, 512))
        for i, core_id in enumerate(core_ids):
            self._needle_masks[i] = data_accessor.get_needle_mask(core_id)
        self._patch_views = PatchView.build_collection_from_images_and_masks(
            self._images,
            window_size=patch_size,
            stride=stride,
            align_to="topright",
            mask_lists=[self._prostate_masks, self._needle_masks],
            thresholds=[prostate_mask_threshold, needle_mask_threshold],
        )

        self._metadata_dicts = []
        for core_id in self.core_ids:
            metadata = (
                self._metadata_table[self._metadata_table.core_id == core_id]
                .iloc[0]
                .to_dict()
            )
            self._metadata_dicts.append(metadata)

        self._indices = []
        for i, pv in enumerate(self._patch_views):
            self._indices.extend([(i, j) for j in range(len(pv))])

        self.transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, j = self._indices[idx]
        pv = self._patch_views[i]

        item = {}
        item["patch"] = pv[j]

        metadata = self._metadata_dicts[i].copy()
        item.update(metadata)

        if self.transform is not None:
            item = self.transform(item)
        return item


class RFPatchesDataset(Dataset):
    _metadata_table = data_accessor.get_metadata_table()

    def __init__(
        self,
        core_ids,
        patch_size_mm,
        patch_stride_mm,
        needle_mask_threshold,
        prostate_mask_threshold,
    ):
        self.core_ids = core_ids
        im_size_mm = 28, 46.06
        im_size_px = data_accessor.get_rf_image(core_ids[0], 0).shape
        self.patch_size_px = int(patch_size_mm[0] * im_size_px[0] / im_size_mm[0]), int(
            patch_size_mm[1] * im_size_px[1] / im_size_mm[1]
        )
        self.patch_stride_px = int(
            patch_stride_mm[0] * im_size_px[0] / im_size_mm[0]
        ), int(patch_stride_mm[1] * im_size_px[1] / im_size_mm[1])

        self._images = [data_accessor.get_rf_image(core_id, 0) for core_id in core_ids]
        self._prostate_masks = [
            data_accessor.get_prostate_mask(core_id) for core_id in core_ids
        ]
        self._needle_masks = [
            data_accessor.get_needle_mask(core_id) for core_id in core_ids
        ]

        self._patch_views = PatchView.build_collection_from_images_and_masks(
            self._images,
            window_size=self.patch_size_px,
            stride=self.patch_stride_px,
            align_to="topright",
            mask_lists=[self._prostate_masks, self._needle_masks],
            thresholds=[prostate_mask_threshold, needle_mask_threshold],
        )
        self._indices = []
        for i, pv in enumerate(self._patch_views):
            self._indices.extend([(i, j) for j in range(len(pv))])

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, j = self._indices[idx]
        metadata = (
            self._metadata_table[self._metadata_table.core_id == self.core_ids[i]]
            .iloc[0]
            .to_dict()
        )
        pv = self._patch_views[i]
        patch = pv[j]

        patch = torch.from_numpy(patch.copy())
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)
        from torchvision.transforms.functional import resize

        patch = resize(patch, (224, 224))
        postition = pv.positions[j]

        return {"patch": patch, **metadata, "position": postition}


class SSLTransform:
    def __call__(self, item):
        patch = item["patch"]
        patch = torch.from_numpy(patch).float() / 255.0
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)

        augs = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        ]
        p1 = T.Compose(augs)(patch)
        p2 = T.Compose(augs)(patch)

        return p1, p2


class Transform:
    def __call__(self, item):
        patch = item["patch"]
        patch = torch.from_numpy(patch).float() / 255.0
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)
        item["patch"] = patch
        return item


if __name__ == "__main__":
    args = parse_args()
    main(args)
