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
from ..utils.image_utils import (
    sliding_window_slice_coordinates,
    convert_physical_coordinate_to_pixel_coordinate,
)
from skimage.transform import resize


DATA_ROOT = os.environ.get("DATA_ROOT")
if DATA_ROOT is None:
    raise ValueError("Environment variable DATA_ROOT must be set")


PATIENT = pd.read_csv(os.path.join(DATA_ROOT, "cores_dataset", "patient.csv"))
CORE = pd.read_csv(os.path.join(DATA_ROOT, "cores_dataset", "core.csv"))
BMODE_DATA_PATH = os.path.join(
    DATA_ROOT,
    "bmode_learning_data",
    "nct",
)


def get_patient_splits_by_fold(fold=0, n_folds=5):
    """returns the list of patient ids for the train, val, and test splits."""
    if n_folds == 5:
        # we manually override the this function and use the csv file
        # because the original code uses a random seed to split the data
        # and we want to be able to reproduce the splits
        table = pd.read_csv(
            os.path.join(DATA_ROOT, "cores_dataset", "5fold_splits.csv")
        )
        train_ids = table[table[f"fold_{fold}"] == "train"].id.values.tolist()
        val_ids = table[table[f"fold_{fold}"] == "val"].id.values.tolist()
        test_ids = table[table[f"fold_{fold}"] == "test"].id.values.tolist()
        return train_ids, val_ids, test_ids

    if fold not in range(n_folds):
        raise ValueError(f"Fold must be in range {n_folds}, but got {fold}")

    table = PATIENT

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for i, (train_idx, test_idx) in enumerate(
        kfold.split(table.index, table["center"])
    ):
        if i == fold:
            train, test = table.iloc[train_idx], table.iloc[test_idx]
            break

    train, val = train_test_split(
        train, test_size=0.2, random_state=0, stratify=train["center"]
    )

    train = train.id.values.tolist()
    val = val.id.values.tolist()
    test = test.id.values.tolist()

    return train, val, test


def get_patient_splits_by_center(leave_out="UVA"): 
    """returns the list of patient ids for the train, val, and test splits."""
    if leave_out not in ['UVA', 'CRCEO', 'PCC', 'PMCC', 'JH']: 
        raise ValueError(f"leave_out must be one of 'UVA', 'CRCEO', 'PCC', 'PMCC', 'JH', but got {leave_out}")

    table = PATIENT

    train = table[table.center != leave_out]
    train, val = train_test_split(
        train, test_size=0.2, random_state=0, stratify=train["center"]
    )
    train = train.id.values.tolist()
    val = val.id.values.tolist()
    test = table[table.center == leave_out].id.values.tolist()

    return train, val, test


def get_core_ids(patient_ids):
    """returns the list of core ids for the given patient ids."""
    return CORE[CORE.patient_id.isin(patient_ids)].id.values.tolist()


def remove_benign_cores_from_positive_patients(core_ids):
    """Returns the list of cores in the given list that are either malignant or from patients with no malignant cores."""
    table = CORE.copy()
    table["positive"] = table.grade.apply(lambda g: 0 if g == "Benign" else 1)
    num_positive_for_patient = table.groupby("patient_id").positive.sum()
    num_positive_for_patient.name = "patients_positive"
    table = table.join(num_positive_for_patient, on="patient_id")
    ALLOWED = table.query("positive == 1 or patients_positive == 0").id.to_list()

    return [core for core in core_ids if core in ALLOWED]


def remove_cores_below_threshold_involvement(core_ids, threshold_pct):
    """Returns the list of cores with at least the given percentage of cancer cells."""
    table = CORE
    ALLOWED = table.query(
        "grade == 'Benign' or pct_cancer >= @threshold_pct"
    ).id.to_list()
    return [core for core in core_ids if core in ALLOWED]


def undersample_benign(cores, seed=0, benign_to_cancer_ratio=1):
    """Returns the list of cores with the same cancer cores and the benign cores undersampled to the given ratio."""

    table = CORE
    benign = table.query('grade == "Benign"').id.to_list()
    cancer = table.query('grade != "Benign"').id.to_list()
    import random

    cores_benign = [core for core in cores if core in benign]
    cores_cancer = [core for core in cores if core in cancer]
    rng = random.Random(seed)
    cores_benign = rng.sample(
        cores_benign, int(len(cores_cancer) * benign_to_cancer_ratio)
    )

    return [core for core in cores if core in cores_benign or core in cores_cancer]


@dataclass
class CohortSelectionOptions:
    min_involvement: float = None
    remove_benign_from_positive_patients: bool = False
    benign_to_cancer_ratio: float = None
    seed: int = 0


@dataclass 
class KFoldCohortSelectionOptions(CohortSelectionOptions):
    fold: int = 0
    n_folds: int = 5


@dataclass 
class LeaveOneCenterOutCohortSelectionOptions(CohortSelectionOptions): 
    leave_out: tp.Literal['UVA', 'CRCEO', 'PCC', 'PMCC', 'JH'] = 'UVA'

    
def select_cohort(
    split: str = "train",
    cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
) -> tp.List[int]:
    """Returns the list of core ids for the given split and cohort selection options."""
    if split not in ["train", "val", "test", "all"]:
        raise ValueError(
            f"Split must be 'train', 'val', 'test' or 'all, but got {split}"
        )

    if isinstance(cohort_selection_options, KFoldCohortSelectionOptions):
        train, val, test = get_patient_splits_by_fold(
            cohort_selection_options.fold, cohort_selection_options.n_folds
        )
    elif isinstance(cohort_selection_options, LeaveOneCenterOutCohortSelectionOptions):
        train, val, test = get_patient_splits_by_center(
            cohort_selection_options.leave_out
        ) 
    else: 
        raise NotImplementedError
            
    match split:
        case "train":
            patient_ids = train
        case "val":
            patient_ids = val
        case "test":
            patient_ids = test
        case "all":
            patient_ids = train + val + test
        case _:
            raise ValueError(
                f"Split must be 'train', 'val', 'test' or 'all, but got {split}"
            )

    core_ids = get_core_ids(patient_ids)
    if cohort_selection_options.remove_benign_from_positive_patients:
        core_ids = remove_benign_cores_from_positive_patients(core_ids)
    if cohort_selection_options.min_involvement is not None:
        core_ids = remove_cores_below_threshold_involvement(
            core_ids, cohort_selection_options.min_involvement
        )
    if cohort_selection_options.benign_to_cancer_ratio is not None:
        core_ids = undersample_benign(
            core_ids,
            cohort_selection_options.seed,
            cohort_selection_options.benign_to_cancer_ratio,
        )

    return core_ids


class ExactNCT2013Cores(Dataset):
    def __init__(
        self,
        split="train",
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
    ):
        super().__init__()
        core_ids = select_cohort(split, cohort_selection_options)
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


class ExactNCT2013BModeImages(ExactNCT2013Cores):
    def __init__(
        self,
        split="train",
        transform=None,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
    ) -> None:
        super().__init__(split=split, cohort_selection_options=cohort_selection_options)
        self.transform = transform
        self.needle_mask = np.load(
            os.path.join(BMODE_DATA_PATH, "needle_mask.npy"), mmap_mode="r"
        )
        self._bmode_data = np.load(
            os.path.join(BMODE_DATA_PATH, "bmode_data.npy"), mmap_mode="r"
        )
        with open(os.path.join(BMODE_DATA_PATH, "core_id_to_idx.json"), "r") as f:
            self._core_id_to_idx = json.load(f)
        self._core_id_to_idx
        self.core_info = self.core_info[
            self.core_info.tag.isin(self._core_id_to_idx.keys())
        ]

    def __len__(self) -> int:
        return len(self.core_info)

    def __getitem__(self, index: int) -> tp.Tuple[tp.Any, tp.Any]:
        out = {}
        core_info = super().__getitem__(index)
        out.update(core_info)
        tag = core_info["tag"]
        data_index = self._core_id_to_idx[tag]
        bmode = self._bmode_data[data_index]
        out["bmode"] = bmode
        out["needle_mask"] = self.needle_mask
        if self.transform is not None:
            out = self.transform(out)
        return out


class ExactNCT2013RFImages(ExactNCT2013Cores):
    def __init__(
        self,
        split="train",
        transform=None,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
        cache=False,
    ):
        super().__init__(split, cohort_selection_options)
        self.needle_mask = np.load(
            os.path.join(BMODE_DATA_PATH, "needle_mask.npy"), mmap_mode="r"
        )
        self.transform = transform
        self.cache = cache 
        self._cache = {}

    def __getitem__(self, index):
        if self.cache and index in self._cache:
            out = self._cache[index].copy()

        else: 
            core_info = super().__getitem__(index)
            tag = core_info["tag"]
            out = {}
            out.update(core_info)
            out["needle_mask"] = self.needle_mask
            out["rf_image"] = np.load(
                os.path.join(DATA_ROOT, "cores_dataset", tag, "image.npy"), 
                mmap_mode="r"
            )

            if self.cache:
                self._cache[index] = out

        if self.transform is not None:
            out = self.transform(out)
        return out


class _ExactNCT2013DatasetWithProstateSegmentation(ABC):
    def __init__(self, dataset: ExactNCT2013Cores, transform=None):
        self.dataset = dataset
        self.transform = transform

        available_masks = [
            id for id in self.dataset.core_info.id.values if self.prostate_mask_available(id)
        ]
        self.dataset.core_info = self.dataset.core_info[
            self.dataset.core_info.id.isin(available_masks)
        ]

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def prostate_mask_available(self, core_id):
        ...

    @abstractmethod
    def prostate_mask(self, core_id):
        ...

    def __getitem__(self, index):
        out = self.dataset[index]
        out["prostate_mask"] = self.prostate_mask(out["id"])
        if self.transform is not None:
            out = self.transform(out)
        return out


class _ExactNCT2013DatasetWithManualProstateSegmentation(
    _ExactNCT2013DatasetWithProstateSegmentation
):
    def prostate_mask_available(self, core_id):
        tag = self.dataset.tag_for_core_id(core_id)
        return os.path.exists(
            os.path.join(DATA_ROOT, "cores_dataset", tag, "prostate_mask.npy")
        )

    def prostate_mask(self, core_id):
        tag = self.dataset.tag_for_core_id(core_id)
        return np.load(
            os.path.join(DATA_ROOT, "cores_dataset", tag, "prostate_mask.npy")
        )


class _ExactNCT2013DatasetWithAutomaticProstateSegmentation(
    _ExactNCT2013DatasetWithProstateSegmentation
):
    def __init__(
        self,
        dataset: ExactNCT2013Cores,
        transform=None,
        masks_dir="/ssd005/projects/exactvu_pca/nct_segmentations_medsam_finetuned_2023-11-10",
    ):
        self.masks_dir = masks_dir
        super().__init__(dataset, transform)
        
    def prostate_mask(self, core_id):
        tag = self.dataset.tag_for_core_id(core_id)
        image = Image.open(os.path.join(self.masks_dir, f"{tag}.png"))
        image = (np.array(image) / 255) > 0.5
        image = image.astype(np.uint8)
        image = np.flip(image, axis=0).copy()
        return image

    def prostate_mask_available(self, core_id):
        tag = self.dataset.tag_for_core_id(core_id)
        return f"{tag}.png" in os.listdir(self.masks_dir)


class ExactNCT2013BmodeImagesWithManualProstateSegmentation(
    _ExactNCT2013DatasetWithManualProstateSegmentation
):
    def __init__(
        self,
        split="train",
        transform=None,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
    ):
        super().__init__(
            ExactNCT2013BModeImages(split, None, cohort_selection_options),
            transform,
        )


class ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
    _ExactNCT2013DatasetWithAutomaticProstateSegmentation
):
    def __init__(
        self,
        split="train",
        transform=None,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
        masks_dir="/ssd005/projects/exactvu_pca/nct_segmentations_medsam_finetuned_2023-11-10",
    ):
        super().__init__(
            ExactNCT2013BModeImages(split, None, cohort_selection_options),
            transform,
            masks_dir,
        )


class ExactNCT2013RFImagesWithManualProstateSegmentation(
    _ExactNCT2013DatasetWithManualProstateSegmentation
):
    def __init__(
        self,
        split="train",
        transform=None,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
    ):
        super().__init__(
            ExactNCT2013RFImages(split, None, cohort_selection_options),
            transform,
        )


class ExactNCT2013RFImagesWithAutomaticProstateSegmentation(
    _ExactNCT2013DatasetWithAutomaticProstateSegmentation
):
    def __init__(
        self,
        split="train",
        transform=None,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
        cache=False,
        masks_dir="/ssd005/projects/exactvu_pca/nct_segmentations_medsam_finetuned_2023-11-10",
    ):
        super().__init__(
            ExactNCT2013RFImages(split, None, cohort_selection_options, cache=cache),
            transform,
            masks_dir,
        )


""" 
def extract_patches(
    image,
    needle_mask,
    prostate_mask,
    image_physical_shape,
    patch_options: PatchOptions = PatchOptions(),
):
    from skimage.util import view_as_windows
    from skimage.transform import resize
    from einops import rearrange

    H_mm, W_mm = image_physical_shape
    H, W = image.shape
    H_px_window, W_px_window = int(patch_options.patch_size_mm[0] * H // H_mm), int(
        patch_options.patch_size_mm[1] * W // W_mm
    )
    H_px_stride, W_px_stride = int(patch_options.patch_strides_mm[0] * H // H_mm), int(
        patch_options.patch_strides_mm[1] * W // W_mm
    )

    needle_mask = resize(needle_mask, (H, W), order=0, anti_aliasing=False)
    prostate_mask = resize(prostate_mask, (H, W), order=0, anti_aliasing=False)

    X, Y = np.meshgrid(np.linspace(0, H_mm, H), np.linspace(0, W_mm, W), indexing="ij")

    image_patches = view_as_windows(
        image, (H_px_window, W_px_window), (H_px_stride, W_px_stride)
    )
    needle_mask_patches = view_as_windows(
        needle_mask, (H_px_window, W_px_window), (H_px_stride, W_px_stride)
    )
    prostate_mask_patches = view_as_windows(
        prostate_mask, (H_px_window, W_px_window), (H_px_stride, W_px_stride)
    )
    X_patches = view_as_windows(
        X, (H_px_window, W_px_window), (H_px_stride, W_px_stride)
    )
    Y_patches = view_as_windows(
        Y, (H_px_window, W_px_window), (H_px_stride, W_px_stride)
    )

    mask = (
        needle_mask_patches.mean((-1, -2)) >= patch_options.needle_mask_threshold
    ) * (prostate_mask_patches.mean((-1, -2)) >= patch_options.needle_mask_threshold)

    mask_flat = rearrange(mask, "h w -> (h w)")
    X_flat = rearrange(X_patches, "nh nw h w -> (nh nw) h w ")
    Y_flat = rearrange(Y_patches, "nh nw h w -> (nh nw) h w")
    image_flat = rearrange(image_patches, "nh nw h w -> (nh nw) h w")

    image_patches = image_flat[mask_flat]
    X_flat = X_flat[mask_flat]
    Y_flat = Y_flat[mask_flat]
    X_min, X_max = X_flat.min(axis=(-1, -2)), X_flat.max(axis=(-1, -2))
    Y_min, Y_max = Y_flat.min(axis=(-1, -2)), Y_flat.max(axis=(-1, -2))
    pos = np.stack([X_min, Y_min, X_max, Y_max], axis=-1)

    return image_patches, pos """


@dataclass(frozen=True)
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


def compute_base_positions(image_physical_size, patch_options):
    axial_slices, lateral_slices = sliding_window_slice_coordinates(
        window_size=patch_options.patch_size_mm,
        strides=patch_options.strides,
        image_size=image_physical_size,
    )
    for i in range(len(axial_slices)):
        for j in range(len(lateral_slices)):
            # positions in xmin_mm, ymin_mm, xmax_mm, ymax_mm
            yield {
                "position": (
                    axial_slices[i][0],
                    lateral_slices[j][0],
                    axial_slices[i][1],
                    lateral_slices[j][1],
                )
            }


def compute_mask_intersections(
    position_data, mask, mask_name, mask_physical_shape, threshold
):
    "position_data is a dictionary with keys 'position'"
    import copy 
    position_data = copy.deepcopy(position_data)

    for position_datum in position_data:
        xmin, ymin, xmax, ymax = position_datum["position"]
        xmin_px, ymin_px = convert_physical_coordinate_to_pixel_coordinate(
            (xmin, ymin), mask_physical_shape, mask.shape
        )
        xmax_px, ymax_px = convert_physical_coordinate_to_pixel_coordinate(
            (xmax, ymax), mask_physical_shape, mask.shape
        )
        mask_patch = mask[xmin_px:xmax_px, ymin_px:ymax_px]
        intersection = np.sum(mask_patch) / mask_patch.size
        position_datum[f"{mask_name}_mask_intersection"] = intersection

        if intersection > threshold:
            yield position_datum


def select_patch(image, position_dict, patch_options):
    position_dict = position_dict.copy()

    xmin_mm, ymin_mm, xmax_mm, ymax_mm = position_dict.pop("position")

    # we shift the patch by a random amount
    xshift_delta = patch_options.shift_delta_mm
    yshift_delta = patch_options.shift_delta_mm
    xshift = np.random.uniform(-xshift_delta, xshift_delta)
    yshift = np.random.uniform(-yshift_delta, yshift_delta)

    if (xmin_mm + xshift) < 0 or (xmax_mm + xshift) > 28:
        xshift = 0

    if (ymin_mm + yshift) < 0 or (ymax_mm + yshift) > 46:
        yshift = 0

    xmin_mm += xshift
    xmax_mm += xshift
    ymin_mm += yshift
    ymax_mm += yshift

    position_dict["position"] = np.array([xmin_mm, ymin_mm, xmax_mm, ymax_mm])

    xmin_px, ymin_px = convert_physical_coordinate_to_pixel_coordinate(
        (xmin_mm, ymin_mm), (28, 46.06), image.shape[:2]
    )
    xmax_px, ymax_px = convert_physical_coordinate_to_pixel_coordinate(
        (xmax_mm, ymax_mm), (28, 46.06), image.shape[:2]
    )

    image_patch = image[xmin_px:xmax_px, ymin_px:ymax_px]

    return image_patch, position_dict


class _ExactNCTPatchesDataset(Dataset):
    def __init__(self, dataset: _ExactNCT2013DatasetWithAutomaticProstateSegmentation, item_name_for_patches, prescale_image=True, transform=None, patch_options: PatchOptions = None):
        super().__init__()
        self.dataset = dataset
        self.item_name_for_patches = item_name_for_patches
        self.transform = transform
        self.patch_options = patch_options
        self.prescale_image = prescale_image

        self.base_positions = list(compute_base_positions((28, 46.06), patch_options))
        _needle_mask = resize(
            self.dataset.dataset.needle_mask, (256, 256), order=0, anti_aliasing=False
        )
        self.base_positions = list(
            compute_mask_intersections(
                self.base_positions,
                _needle_mask,
                "needle",
                (28, 46.06),
                patch_options.needle_mask_threshold,
            )
        )
        self.positions = []
        for i in tqdm(range(len(self.dataset)), desc="Computing positions"):
            positions = self.base_positions.copy()
            positions = list(
                compute_mask_intersections(
                    positions,
                    self.dataset[i]["prostate_mask"],
                    "prostate",
                    (28, 46.06),
                    patch_options.prostate_mask_threshold,
                )
            )
            self.positions.append(positions)
        self._indices = []
        for i in range(len(self.dataset)):
            for j in range(len(self.positions[i])):
                self._indices.append((i, j))

    def __getitem__(self, index):
        i, j = self._indices[index]
        item = self.dataset[i]

        image = item.pop(self.item_name_for_patches)
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


class ExactNCT2013BmodePatches(_ExactNCTPatchesDataset):
    def __init__(
        self,
        split="train",
        transform=None,
        prescale_image: bool = False,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
        patch_options: PatchOptions = PatchOptions(),
    ):
        dataset = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split, transform=None, cohort_selection_options=cohort_selection_options
        )
        super().__init__(
            dataset,
            "bmode",
            prescale_image=prescale_image,
            transform=transform,
            patch_options=patch_options,
        )


class ExactNCT2013RFImagePatches(_ExactNCTPatchesDataset):
    def __init__(
        self,
        split="train",
        transform=None,
        prescale_image: bool = False,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
        patch_options: PatchOptions = PatchOptions(),
    ):
        dataset = ExactNCT2013RFImagesWithAutomaticProstateSegmentation(
            split, transform=None, cohort_selection_options=cohort_selection_options, cache=True
        )
        super().__init__(
            dataset,
            "rf_image",
            prescale_image=prescale_image,
            transform=transform,
            patch_options=patch_options,
        )



""" 

class ExactNCT2013BmodePatches(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        prescale_image: bool = False,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
        patch_options: PatchOptions = PatchOptions(),
    ):
        super().__init__()
        self.patch_options = patch_options
        self.transform = transform
        self.prescale_image = prescale_image
        self.split = split
        self.dataset = ExactNCT2013BmodeImagesWithAutomaticProstateSegmentation(
            split, transform=None, cohort_selection_options=cohort_selection_options
        )

        self.base_positions = list(compute_base_positions((28, 46.06), patch_options))
        _needle_mask = resize(
            self.dataset.dataset.needle_mask, (256, 256), order=0, anti_aliasing=False
        )
        self.base_positions = list(
            compute_mask_intersections(
                self.base_positions,
                _needle_mask,
                "needle",
                (28, 46.06),
                patch_options.needle_mask_threshold,
            )
        )
        self.positions = []
        for i in tqdm(range(len(self.dataset)), desc="Computing positions"):
            positions = self.base_positions.copy()
            positions = list(
                compute_mask_intersections(
                    positions,
                    self.dataset[i]["prostate_mask"],
                    "prostate",
                    (28, 46.06),
                    patch_options.prostate_mask_threshold,
                )
            )
            self.positions.append(positions)
        self._indices = []
        for i in range(len(self.dataset)):
            for j in range(len(self.positions[i])):
                self._indices.append((i, j))

    def __getitem__(self, index):
        i, j = self._indices[index]
        item = self.dataset[i]
        image = item.pop("bmode")
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


 """