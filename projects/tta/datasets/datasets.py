# from sklearn.model_selection import StratifiedKFold, train_test_split
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
from medAI.utils.image_utils import (
    sliding_window_slice_coordinates,
    convert_physical_coordinate_to_pixel_coordinate,
)
from skimage.transform import resize

from medAI.datasets.nct2013 import *


RF_DATA_PATH = os.path.join(DATA_ROOT, "cores_dataset")

@dataclass
class CohortSelectionOptions:
    fold: int = 0
    n_folds: int = 5
    min_involvement: float = None
    remove_benign_from_positive_patients: bool = False
    benign_to_cancer_ratio: float = None
    seed: int = 0

@dataclass
class PatchOptions:
    """Options for generating a set of patches from a core."""

    patch_size_mm: tp.Tuple[float, float] = (5, 5)
    strides: tp.Tuple[float, float] = (
        1,
        1,
    )  # defines the stride in mm of the base positions
    needle_mask_threshold: float = 0.5  # if not None, then only positions with a needle mask intersection greater than this value are kept
    prostate_mask_threshold: float = -1
    shift_delta_mm: float = 0.0  # whether to randomly shift the patch by a small amount
    # output_size_px: tp.Tuple[int, int] | None = None # if not None, then the patch is resized to this size in pixels


class ExactNCT2013RFImagesWithProstateSegmenation(ExactNCT2013RFImages, ABC):
    def __init__(
        self,
        split="train",
        transform=None,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
    ):
        super().__init__(
            split,
            transform=None,
            cohort_selection_options=cohort_selection_options,
        )
        self.transform = transform
        available_masks = [
            id for id in self.core_info.id.values if self.prostate_mask_available(id)
        ]
        self.core_info = self.core_info[self.core_info.id.isin(available_masks)]

    @abstractmethod
    def prostate_mask_available(self, core_id):
        ...

    @abstractmethod
    def prostate_mask(self, core_id):
        ...

    def __getitem__(self, index: int) -> tp.Tuple[tp.Any, tp.Any]:
        tmp_transform = self.transform
        self.transform = None
        out = super().__getitem__(index)
        self.transform = tmp_transform
        out["prostate_mask"] = self.prostate_mask(out["id"])
        if self.transform is not None:
            out = self.transform(out)
        return out


class ExactNCT2013RFImagesWithManualProstateSegmenation(
    ExactNCT2013RFImagesWithProstateSegmenation
):
    def prostate_mask_available(self, core_id):
        tag = self.tag_for_core_id(core_id)
        return os.path.exists(
            os.path.join(DATA_ROOT, "cores_dataset", tag, "prostate_mask.npy")
        )

    def prostate_mask(self, core_id):
        tag = self.tag_for_core_id(core_id)
        return np.load(
            os.path.join(DATA_ROOT, "cores_dataset", tag, "prostate_mask.npy")
        )


class ExactNCT2013RFImagesWithAutomaticProstateSegmentation(
    ExactNCT2013RFImagesWithProstateSegmenation
):
    def __init__(
        self,
        split="train",
        transform=None,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
        masks_dir="/ssd005/projects/exactvu_pca/nct_segmentations_medsam_finetuned_2023-11-10",
    ):
        self.masks_dir = masks_dir
        super().__init__(split, transform, cohort_selection_options)

    def prostate_mask(self, core_id):
        tag = self.tag_for_core_id(core_id)
        image = Image.open(os.path.join(self.masks_dir, f"{tag}.png"))
        image = (np.array(image) / 255) > 0.5
        image = image.astype(np.uint8)
        image = np.flip(image, axis=0).copy()
        return image

    def prostate_mask_available(self, core_id):
        tag = self.tag_for_core_id(core_id)
        return f"{tag}.png" in os.listdir(self.masks_dir)


class ExactNCT2013RFPatches(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        prescale_image: bool = False,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
        patch_options: PatchOptions = PatchOptions(),
        debug: bool = False,
    ):
        super().__init__()
        self.patch_options = patch_options
        self.transform = transform
        self.prescale_image = prescale_image
        self.split = split
        self.dataset = ExactNCT2013RFImagesWithAutomaticProstateSegmentation(
            split, transform=None, cohort_selection_options=cohort_selection_options
        )

        self.base_positions = list(compute_base_positions(
            (28, 46.06), patch_options
        )) 
        _needle_mask = resize(self.dataset.needle_mask, (256, 256), order=0, anti_aliasing=False)
        self.base_positions = list(compute_mask_intersections(
            self.base_positions,
            _needle_mask,
            "needle",
            (28, 46.06),
            patch_options.needle_mask_threshold,
        ))
        self.positions = [] 
        for i in tqdm(range(len(self.dataset)), desc=f"Computing positions {split}"): 
            positions = self.base_positions.copy()
            positions = list(compute_mask_intersections(
                positions,
                self.dataset[i]["prostate_mask"],
                "prostate",
                (28, 46.06),
                patch_options.prostate_mask_threshold,
            ))
            self.positions.append(positions)
            
            if debug and i > 100:
                break 
            
        self._indices = []
        for i in range(len(self.dataset)):
            for j in range(len(self.positions[i])):
                self._indices.append((i, j))
            
            if debug and i > 100:
                break

    def __getitem__(self, index):
        i, j = self._indices[index]
        item = self.dataset[i]
        image = item.pop("rf_image")
        if self.prescale_image: 
            image = (image - image.min()) / (image.max() - image.min())

        item.pop("needle_mask")
        item.pop("prostate_mask")

        item["label"] = item["grade"] != "Benign"

        position = self.positions[i][j]

        image_patch, position = select_patch(image, position, self.patch_options)

        item["patch"] = image_patch
        item.update(position)

        if self.transform is not None:
            item = self.transform(item)

        return item
    
    def __len__(self):
        return len(self._indices)


@dataclass
class SupportPatchConfig:
    num_support_patches: int = 10
    
    
class ExactNCT2013RFPatchesWithSupportPatches(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        prescale_image: bool = False,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
        patch_options: PatchOptions = PatchOptions(),
        support_patch_config: SupportPatchConfig = SupportPatchConfig(),
        debug: bool = False,
    ):
        super().__init__()
        self.patch_options = patch_options
        self.transform = transform
        self.prescale_image = prescale_image
        self.split = split
        self.support_patch_config = support_patch_config
        self.dataset = ExactNCT2013RFImagesWithAutomaticProstateSegmentation(
            split, transform=None, cohort_selection_options=cohort_selection_options
        )

        self.base_positions = list(compute_base_positions(
            (28, 46.06), patch_options
        )) 
        _needle_mask = resize(self.dataset.needle_mask, (256, 256), order=0, anti_aliasing=False)
        
        self.support_positions = []
        self.query_positions = [] 
        for i in tqdm(range(len(self.dataset)), desc=f"Computing positions {split}"): 
            # updates self.query_positions and self.support_positions
            (support_possitions,
             query_positions
             ) = self.compute_masks_intersections_update_positions(
                 self.base_positions,
                 self.dataset[i]["prostate_mask"],
                 _needle_mask,
                 "prostate_needle",
                 (28, 46.06),
                 patch_options.prostate_mask_threshold,
                 patch_options.needle_mask_threshold,
                 )
            self.support_positions.append(support_possitions)
            self.query_positions.append(query_positions)
            
            if debug and i > 10:
                break 
            
        self._indices = []
        for i in range(len(self.dataset)):
            for j in range(len(self.query_positions[i])):
                self._indices.append((i, j))
            
            if debug and i > 10:
                break
                
    def compute_masks_intersections_update_positions(
        self,
        position_data,
        prostate_mask,
        needle_mask,
        mask_name,
        mask_physical_shape,
        prostate_threshold,
        needle_threshold
    ):
        "position_data is a dictionary with keys 'position'"

        support_positions = []
        query_positions = []        
        for position_datum in position_data:
            xmin, ymin, xmax, ymax = position_datum["position"]
            
            # prostate_mask intersection        
            xmin_px, ymin_px = convert_physical_coordinate_to_pixel_coordinate(
                (xmin, ymin), mask_physical_shape, prostate_mask.shape
            )
            xmax_px, ymax_px = convert_physical_coordinate_to_pixel_coordinate(
                (xmax, ymax), mask_physical_shape, prostate_mask.shape
            )
            mask_patch = prostate_mask[xmin_px:xmax_px, ymin_px:ymax_px]
            intersection1 = np.sum(mask_patch) / mask_patch.size

            # needle_mask intersection
            xmin_px, ymin_px = convert_physical_coordinate_to_pixel_coordinate(
                (xmin, ymin), mask_physical_shape, needle_mask.shape
            )
            xmax_px, ymax_px = convert_physical_coordinate_to_pixel_coordinate(
                (xmax, ymax), mask_physical_shape, needle_mask.shape
            )
            mask_patch = needle_mask[xmin_px:xmax_px, ymin_px:ymax_px]
            intersection2 = np.sum(mask_patch) / mask_patch.size
            
            # update position_datum
            if intersection1 > prostate_threshold:
                if intersection2 < needle_threshold:
                    support_positions.append(position_datum)
                else:
                    query_positions.append(position_datum)
            
        return support_positions, query_positions

    def __getitem__(self, index):
        i, j = self._indices[index]
        item = self.dataset[i]
        image = item.pop("rf_image")
        if self.prescale_image: 
            image = (image - image.min()) / (image.max() - image.min())

        item.pop("needle_mask")
        item.pop("prostate_mask")

        item["label"] = item["grade"] != "Benign"

        position = self.query_positions[i][j]
               
        image_patch, position = select_patch(image, position, self.patch_options)

        item["patch"] = image_patch
        item.update(position)

        # Randomly add num_support_patches of support patches 
        support_patches = []
        support_positions = self.support_positions[i]
        support_positions = np.random.choice(support_positions, self.support_patch_config.num_support_patches)
        min_axial, min_lateral = np.inf, np.inf
        for support_position in support_positions:
            support_patch, _ = select_patch(image, support_position, self.patch_options)
            support_patches.append(support_patch)
            min_axial = min(min_axial, support_patch.shape[0])
            min_lateral = min(min_lateral, support_patch.shape[1])
        
        # Resize support patches to the same size and stack them
        support_patches = [support_patch[:min_axial, :min_lateral] for support_patch in support_patches]
        item["support_patches"] = np.stack(support_patches)
        
        if self.transform is not None:
            item = self.transform(item)

        return item
    
    def __len__(self):
        return len(self._indices)
    
