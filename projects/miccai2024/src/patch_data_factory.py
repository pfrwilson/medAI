import logging
import os
from argparse import ArgumentParser
from typing import Any

import numpy as np
import torch
import wandb
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
from medAI.utils.data.patch_extraction import PatchView


class PatchDataFactory:
    def __init__(self, test_center, val_seed, patch_size, stride, batch_size):
        self.test_center = test_center
        self.val_seed = val_seed
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size

        

    @classmethod
    def add_args(cls, parser): 
        group = parser.add_argument_group(
        "Data", "Arguments for building the BMode patch dataloaders"
        )
        group.add_argument("--test_center", type=str, default="UVA")
        group.add_argument("--val_seed", type=int, default=0)
        group.add_argument("--patch_size", type=int, default=128)
        group.add_argument("--stride", type=int, default=32)
        group.add_argument("--batch_size", type=int, default=128)

        return parser

    @classmethod
    def from_args(cls, args):
        return cls(
            args.test_center,
            args.val_seed,
            args.patch_size,
            args.stride,
            args.batch_size,
        )

    def __call__(self): 
        ...


def add_args(parser, include_unlabeled=True):
    group = parser.add_argument_group(
        "Data", "Arguments for building the BMode patch dataloaders"
    )
    group.add_argument("--test_center", type=str, default="UVA")
    group.add_argument("--val_seed", type=int, default=0)
    group.add_argument("--patch_size", type=int, default=128)
    group.add_argument("--stride", type=int, default=32)
    group.add_argument("--batch_size", type=int, default=128)
    group.add_argument(
        "--full_prostate",
        action="store_true",
        help="""(Only relevant for the unlabeled dataset) If set, take unlabeled patches from the whole prostate, 
                       not just the needle region. This could improve the performance of the SSL model.""",
    )
    return parser


def make_data_loaders_from_args(args, include_unlabeled=True):
    return make_data_loaders(
        args.test_center,
        args.val_seed,
        args.patch_size,
        args.stride,
        args.batch_size,
        args.full_prostate,
        include_unlabeled=include_unlabeled,
    )


def make_data_loaders(
    test_center,
    val_seed,
    patch_size,
    stride,
    batch_size,
    full_prostate,
    include_unlabeled=True,
):
    print(f"Preparing data loaders for test center {test_center}")

    train_patients, val_patients, test_patients = get_patient_splits_by_center(
        test_center, val_size=0.2, val_seed=val_seed
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

    if include_unlabeled:
        print("SSL dataset...")
        ssl_dataset = BModePatchesDataset(
            ssl_train_core_ids,
            patch_size=(patch_size, patch_size),
            stride=(stride, stride),
            needle_mask_threshold=0.6 if not full_prostate else -1,
            prostate_mask_threshold=-1 if not full_prostate else 0.1,
            transform=SSLTransform(),
        )
    print("Train dataset...")
    train_dataset = BModePatchesDataset(
        train_core_ids,
        patch_size=(patch_size, patch_size),
        stride=(stride, stride),
        needle_mask_threshold=0.6,
        prostate_mask_threshold=-1,
        transform=Transform(),
    )
    print("Val dataset...")
    val_dataset = BModePatchesDataset(
        val_core_ids,
        patch_size=(patch_size, patch_size),
        stride=(stride, stride),
        needle_mask_threshold=0.6,
        prostate_mask_threshold=-1,
        transform=Transform(),
    )
    print("Test dataset...")
    test_dataset = BModePatchesDataset(
        test_core_ids,
        patch_size=(patch_size, patch_size),
        stride=(stride, stride),
        needle_mask_threshold=0.6,
        prostate_mask_threshold=-1,
        transform=Transform(),
    )

    if include_unlabeled:
        ssl_loader = torch.utils.data.DataLoader(
            ssl_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"SSL Train batches: {len(ssl_loader)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    if include_unlabeled:
        return ssl_loader, train_loader, val_loader, test_loader
    else:
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
