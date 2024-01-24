"""
A dataset implementation for the NCT 2013 B-mode dataset, where the bmode images 
have been resized to 1024x1024px and where patch extraction logic uses pixel sizes

The benefit of this dataset is that the patch dataset is easier to relate back to
the image dataset, so that patch-based models can be more easily compared to image-based 
models. 
"""


from typing import Any
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import pandas as pd
import typing as tp
from torch.utils.data import Dataset
import numpy as np
import json
from dataclasses import dataclass
from PIL import Image
from abc import ABC, abstractmethod
from tqdm import tqdm
from skimage.transform import resize
from diskcache import Cache
from einops import rearrange
import matplotlib.pyplot as plt

DATA_ROOT = os.environ.get("DATA_ROOT")
if DATA_ROOT is None:
    raise ValueError("Environment variable DATA_ROOT must be set")

DATASET_KEY = "2023-12-14_bmode_dataset_1024px"

from .nct2013 import (
    CoresSelector,
    KFoldPatientSelector,
    LeaveOneCenterOutPatientSelector,
    CORE,
    PATIENT,
    PatientSelector,
)


class CoreInfo(Dataset):
    def __init__(
        self,
        split="train",
        cores_selector: CoresSelector = CoresSelector(),
    ):
        super().__init__()
        core_ids = cores_selector(split)
        self.core_info = (
            CORE.set_index("patient_id")
            .join(PATIENT.rename(columns={"id": "patient_id"}).set_index("patient_id"))
            .reset_index()
        )
        self.core_info = self.core_info[self.core_info.id.isin(core_ids)]

    def __getitem__(self, index):
        out = {}
        core_info = dict(self.core_info.iloc[index])
        out.update(core_info)
        return out

    def __len__(self):
        return len(self.core_info)

    def tag_for_core_id(self, core_id):
        return self.core_info[self.core_info.id == core_id].tag.values[0]


def load_prostate_mask(tag):
    from PIL import Image

    mask = Image.open(
        os.path.join(DATA_ROOT, DATASET_KEY, "prostate_masks", f"{tag}.png")
    )
    mask = np.array(mask)
    mask = np.flip(mask, axis=0).copy()
    mask = mask / 255
    return mask


_needle_mask = None


def load_needle_mask():
    global _needle_mask
    if _needle_mask is None:
        _needle_mask = np.load(os.path.join(DATA_ROOT, DATASET_KEY, "needle_mask.npy"))
    return _needle_mask


class BModeImages(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        cores_selector: CoresSelector = CoresSelector(),
    ):
        super().__init__()
        self.core_info = CoreInfo(split, cores_selector)
        self.transform = transform

        self._bmode_data = np.load(
            os.path.join(DATA_ROOT, DATASET_KEY, "bmode_data.npy"), mmap_mode="r"
        )
        import json

        self._core_tag_to_index = json.load(
            open(os.path.join(DATA_ROOT, DATASET_KEY, "core_id_to_idx.json"))
        )
        self.needle_mask = load_needle_mask()

    def __getitem__(self, idx):
        info = self.core_info[idx]
        tag = info["tag"]

        bmode = self._bmode_data[self._core_tag_to_index[tag]]
        needle_mask = self.needle_mask
        prostate_mask = load_prostate_mask(tag)

        item = {
            "bmode": bmode,
            "needle_mask": needle_mask,
            "prostate_mask": prostate_mask,
            **info,
        }

        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.core_info)


def n_windows(image_size, patch_size, stride):
    return 1 + (image_size - patch_size) // stride


def valid_patch_positions(
    tag, patch_size, stride, needle_mask_threshold, prostate_mask_threshold
):
    positions = np.mgrid[
        0 : n_windows(1024, patch_size, stride) * stride : stride,
        0 : n_windows(1024, patch_size, stride) * stride : stride,
    ]
    positions = rearrange(positions, "c h w -> (h w) c")
    positions = np.concatenate([positions, positions + patch_size], axis=-1)
    needle_mask = load_needle_mask()
    needle_mask = resize(needle_mask, (1024, 1024), order=0)
    from skimage.util import view_as_windows

    needle_mask = view_as_windows(needle_mask, (patch_size, patch_size), stride)
    needle_mask = needle_mask.mean(axis=(2, 3)) > needle_mask_threshold
    needle_mask = rearrange(needle_mask, "h w -> (h w)")
    prostate_mask = load_prostate_mask(tag)
    prostate_mask = resize(prostate_mask, (1024, 1024), order=0)
    prostate_mask = view_as_windows(prostate_mask, (patch_size, patch_size), stride)
    prostate_mask = prostate_mask.mean(axis=(2, 3)) > prostate_mask_threshold
    prostate_mask = rearrange(prostate_mask, "h w -> (h w)")
    valid = needle_mask & prostate_mask

    return positions[valid]


class BModePatches(Dataset):
    def __init__(
        self,
        split="train",
        cores_selector: CoresSelector = CoresSelector(),
        patch_size=128,
        stride=128,
        needle_mask_threshold=-1,
        prostate_mask_threshold=-1,
        transform=None,
    ):
        super().__init__()
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        self.core_info = CoreInfo(split, cores_selector)
        self.bmode_images = BModeImages(split, cores_selector=cores_selector)

        # computing patch positions
        position_candidates = np.mgrid[
            0 : n_windows(1024, patch_size, stride) * stride : stride,
            0 : n_windows(1024, patch_size, stride) * stride : stride,
        ]
        position_candidates = rearrange(position_candidates, "c h w -> (h w) c")
        position_candidates = np.concatenate(
            [position_candidates, position_candidates + patch_size], axis=-1
        )

        # since it is the same for every image, we can apply the needle mask threshold here
        needle_mask = load_needle_mask()
        needle_mask = resize(needle_mask, (1024, 1024), order=0)

        new_position_candidates = []
        for position_candidate in tqdm(
            position_candidates, desc="Applying needle mask"
        ):
            x1, y1, x2, y2 = position_candidate
            patch = needle_mask[x1:x2, y1:y2]
            if patch.mean() > needle_mask_threshold:
                new_position_candidates.append(position_candidate)
        position_candidates = np.array(new_position_candidates)

        # loading all prostate masks
        prostate_masks = []
        for idx in tqdm(range(len(self.core_info)), desc="Loading Prostate Masks"):
            tag = self.core_info[idx]["tag"]
            prostate_mask = load_prostate_mask(tag)
            prostate_masks.append(prostate_mask)
        prostate_masks = np.stack(prostate_masks, axis=-1)

        n_images = len(self.core_info)
        n_position_candidates = len(position_candidates)
        valid_position_candidates = np.zeros(
            (n_images, n_position_candidates), dtype=bool
        )

        for idx in tqdm(range(n_position_candidates), desc="Applying prostate mask"):
            x1, y1, x2, y2 = position_candidates[idx]
            x1 = int(x1 / 1024 * prostate_masks.shape[0])
            x2 = int(x2 / 1024 * prostate_masks.shape[0])
            y1 = int(y1 / 1024 * prostate_masks.shape[1])
            y2 = int(y2 / 1024 * prostate_masks.shape[1])

            valid_position_candidates[:, idx] = (
                prostate_masks[x1:x2, y1:y2].mean(axis=(0, 1)) > prostate_mask_threshold
            )

        self._indices = []
        self._positions = []
        for idx in tqdm(range(n_images), desc="Filtering positions"):
            positions_for_core = []
            for j in range(n_position_candidates):
                if valid_position_candidates[idx, j]:
                    position = position_candidates[j]
                    positions_for_core.append(position)
                    self._indices.append((idx, len(positions_for_core) - 1))
            self._positions.append(positions_for_core)

    def __getitem__(self, idx):
        core_idx, patch_idx = self._indices[idx]
        info = self.core_info[core_idx]

        bmode = self.bmode_images[core_idx]["bmode"]

        positions = self._positions[core_idx][patch_idx]
        x1, y1, x2, y2 = positions
        patch = bmode[x1:x2, y1:y2]

        item = {
            "patch": patch,
            "position": positions,
            **info,
        }

        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self._indices)

    @property
    def num_cores(self):
        return len(self.core_info)

    def list_patches_for_core(self, core_idx):
        indices = [i for i, (idx, _) in enumerate(self._indices) if idx == core_idx]
        return [self._indices[i] for i in indices]

    def show_patch_extraction(self, ax=None):
        """Illustrates the patch extraction process for a random core

        Shows a random image from the dataset, with the extracted patches
        highlighted with red transparent rectangles.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
        """
        ax = ax or plt.gca()

        import matplotlib.patches as patches
        import random

        core_idx = random.randint(0, len(self.core_info) - 1)
        info = self.core_info[core_idx]
        tag = info["tag"]
        bmode = self.bmode_images[core_idx]["bmode"]
        positions = self._positions[core_idx]
        ax.imshow(bmode, cmap="gray")
        for position in positions:
            x1, y1, x2, y2 = position
            rect = patches.Rectangle(
                (y1, x1),
                y2 - y1,
                x2 - x1,
                linewidth=1,
                edgecolor="black",
                facecolor="red",
                alpha=0.1,
            )
            ax.add_patch(rect)
        ax.axis("off")
