import torch
import numpy as np
import logging
import os
import pandas as pd
import typing as tp
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from src.image_utils import (
    sliding_window_slice_coordinates,
    convert_physical_coordinate_to_pixel_coordinate,
)
from src.cohort_selection import generate_splits, SplitsConfig
import matplotlib.pyplot as plt
import copy


PROSTATE_MASKS_DIR = "/ssd005/projects/exactvu_pca/nct_segmentations"
BMODE_DATA_DIR = "/ssd005/projects/exactvu_pca/bmode_learning_data/nct"
RF_DATA_DIR = "/ssd005/projects/exactvu_pca/cores_dataset"


def lookup_prostate_mask(specifier):
    fpath = os.path.join(PROSTATE_MASKS_DIR, f"{specifier}.png")
    from PIL import Image

    mask = np.array(Image.open(fpath)) / 255.0
    mask = np.flip(mask, axis=0)
    return mask


def create_bmode_colormap():
    import numpy as np
    import matplotlib.pyplot as plt

    g7 = np.load("/h/pwilson/projects/bmode_learning/data/G7.npy")
    # the entries of g7 are RGB values for the corresponding uint8 pixel value.
    # make a colormap from this
    from matplotlib.colors import ListedColormap

    g7_cmap = ListedColormap(g7 / 255)

    return g7_cmap


class RFData:
    _RF_DATA = None
    _RF_CORE_ID_TO_IDX = None

    @classmethod
    def get_rf(cls, core_id):
        if cls._RF_DATA is None:
            cls._RF_DATA = np.load(
                os.path.join(RF_DATA_DIR, "rf_data.npy"), mmap_mode="r"
            )
            cls._RF_CORE_ID_TO_IDX = {
                core_id: i
                for i, core_id in enumerate(
                    np.load(os.path.join(RF_DATA_DIR, "core_specifiers.npy"))
                )
            }
        return cls._RF_DATA[cls._RF_CORE_ID_TO_IDX[core_id]]


class UndersampledBModeFinder:
    _data = np.load(
        "/ssd005/projects/exactvu_pca/bmode_learning_data/nct/bmode_data.npy",
        mmap_mode="r",
    )
    import json

    with open(
        "/ssd005/projects/exactvu_pca/bmode_learning_data/nct/core_id_to_idx.json"
    ) as f:
        _core_id_to_idx = json.load(f)
    idx_to_core_id = {v: k for k, v in _core_id_to_idx.items()}

    @classmethod
    def get_image(cls, core_id):
        idx = cls._core_id_to_idx[core_id]
        return cls._data[idx]


def needle_mask():
    return np.load(os.path.join(BMODE_DATA_DIR, "needle_mask.npy"))


class ImageFinderV2:
    def __init__(self, image_type):
        self.mode = image_type
        self._data = {}

    def _get_rf(self, core_id):
        return RFData.get_rf(core_id)

    def get_image(self, core_id):
        if core_id not in self._data:
            self._data[core_id] = self._load_image(core_id)
        return self._data[core_id]

    def _load_image(self, core_id):
        if self.mode == "bmode":
            return np.load(os.path.join(RF_DATA_DIR, core_id, "bmode.npy"))
        elif self.mode == "rf":
            return self._get_rf(core_id)
        elif self.mode == "split_channel_image":
            rf = self._get_rf(core_id)
            from scipy.signal import hilbert

            rf_hilbert = hilbert(rf)
            rf_envelope = np.abs(rf_hilbert)
            rf_theta = np.angle(rf_hilbert)
            return np.stack([rf_envelope, rf_theta], axis=-1)
        elif self.mode == "envelope":
            rf = self._get_rf(core_id)
            from scipy.signal import hilbert

            rf_hilbert = hilbert(rf)
            rf_envelope = np.abs(rf_hilbert)
            return rf_envelope
        elif self.mode == "theta":
            rf = self._get_rf(core_id)
            from scipy.signal import hilbert

            rf_hilbert = hilbert(rf)
            rf_theta = np.angle(rf_hilbert)
            return rf_theta
        elif self.mode == "bmode_undersampled":
            return UndersampledBModeFinder.get_image(core_id)
        else:
            raise ValueError(f"Unsupported mode {self.mode}")


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


def get_metadata_for_id(core_id):
    metadata = pd.read_csv(os.path.join(BMODE_DATA_DIR, "metadata.csv"))
    out = dict(metadata[metadata["core_id"] == core_id].iloc[0])
    out["core_specifier"] = out["core_id"]
    return out


def default_patch_transform(patch):
    import cv2

    patch = cv2.resize(patch, (256, 256))
    patch = (patch - patch.mean()) / patch.std()
    patch = torch.from_numpy(patch).float()
    patch = patch.unsqueeze(0)

    return patch


def default_label_transform(label):
    return torch.tensor(label, dtype=torch.int64)


def shift_patch_position(position, shift_delta_mm):
    xmin_mm, ymin_mm, xmax_mm, ymax_mm = position

    # we shift the patch by a random amount
    xshift_delta = shift_delta_mm
    yshift_delta = shift_delta_mm
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

    return xmin_mm, ymin_mm, xmax_mm, ymax_mm


class PatchesDatasetV3(torch.utils.data.Dataset):
    def __init__(
        self,
        core_ids,
        mode: tp.Literal[
            "bmode",
            "rf",
            "split_channel_image",
            "envelope",
            "theta",
            "bmode_undersampled",
        ] = "bmode",
        transform=default_patch_transform,
        target_transform=default_label_transform,
        patch_options: PatchOptions = PatchOptions(),
    ):
        super().__init__()

        self.core_ids = core_ids
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.patch_options = patch_options

        self._image_finder = ImageFinderV2(mode)
        self._bmode_finder = ImageFinderV2("bmode")

        self.metadata = pd.read_csv(os.path.join(BMODE_DATA_DIR, "metadata.csv"))

        failed_cores = []
        # self.cores = {}
        for core_id in self.core_ids:
            try:
                self._image_finder.get_image(core_id)
            except:
                failed_cores.append(core_id)
        self.core_ids = [
            core_id for core_id in self.core_ids if core_id not in failed_cores
        ]
        print("Failed cores: ", failed_cores)

        from trusnet.utils.image_utils import (
            sliding_window_slice_coordinates,
        )

        axial_slices, lateral_slices = sliding_window_slice_coordinates(
            window_size=self.patch_options.patch_size_mm,
            strides=self.patch_options.strides,
            image_size=(28, 46),
        )
        base_positions = []
        for i in range(len(axial_slices)):
            for j in range(len(lateral_slices)):
                # positions in xmin_mm, ymin_mm, xmax_mm, ymax_mm
                base_positions.append(
                    {
                        "position": (
                            axial_slices[i][0],
                            lateral_slices[j][0],
                            axial_slices[i][1],
                            lateral_slices[j][1],
                        )
                    }
                )

        # Since the needle mask is fixed we can filter by the needle mask just once
        if self.patch_options.needle_mask_threshold is not None:
            needle_mask = np.load(os.path.join(BMODE_DATA_DIR, "needle_mask.npy"))
            base_positions = self.filter_positions_by_mask(
                base_positions,
                needle_mask,
                "needle",
                threshold=self.patch_options.needle_mask_threshold,
            )

        positions = {}
        for core_id in tqdm(self.core_ids, desc="Loading and indexing patch positions"):
            if self.patch_options.prostate_mask_threshold is not None:
                try:
                    prostate_mask = lookup_prostate_mask(core_id)
                    positions[core_id] = self.filter_positions_by_mask(
                        base_positions,
                        prostate_mask,
                        "prostate",
                        threshold=self.patch_options.prostate_mask_threshold,
                    )
                except:
                    # if there's no prostate mask we just ignore it
                    positions[core_id] = base_positions
            else:
                positions[core_id] = base_positions

        self.positions = positions
        indices = []
        for core_id in self.core_ids:
            for j in range(len(self.positions[core_id])):
                indices.append((core_id, j))

        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        core_id, position_index = self.indices[index]

        position = self.positions[core_id][position_index].copy()
        core_metadata = self.get_metadata_for_id(core_id)
        core_metadata["label"] = core_metadata["grade"] != "Benign"

        xmin_mm, ymin_mm, xmax_mm, ymax_mm = position.pop("position")

        image = self._image_finder.get_image(core_id)

        # we shift the patch by a random amount
        xshift_delta = self.patch_options.shift_delta_mm
        yshift_delta = self.patch_options.shift_delta_mm
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

        # position["position"] = (xmin_mm, ymin_mm, xmax_mm, ymax_mm)

        xmin_px, ymin_px = convert_physical_coordinate_to_pixel_coordinate(
            (xmin_mm, ymin_mm), (28, 46.06), image.shape[:2]
        )

        xmax_px, ymax_px = convert_physical_coordinate_to_pixel_coordinate(
            (xmax_mm, ymax_mm), (28, 46.06), image.shape[:2]
        )

        # snap the pixel coordinates to the nearest pixel value
        delta_x = xmin_px - round(xmin_px)
        delta_y = ymin_px - round(ymin_px)
        xmin_px -= delta_x
        xmin_px = int(round(xmin_px))
        ymin_px -= delta_y
        ymin_px = int(round(ymin_px))
        xmax_px -= delta_x
        xmax_px = int(round(xmax_px))
        ymax_px -= delta_y
        ymax_px = int(round(ymax_px))

        image_patch = image[xmin_px:xmax_px, ymin_px:ymax_px]

        sample = {
            "patch": image_patch,
            "global_patch_index": index,
            **core_metadata,
            "position": np.array((xmin_mm, ymin_mm, xmax_mm, ymax_mm)),
        }

        if self.transform is not None:
            sample["patch"] = self.transform(sample["patch"])

        if self.target_transform is not None:
            sample["label"] = self.target_transform(sample["label"])

        return sample

    def get_bmode(self, patch_idx=None, core_idx=None):
        if patch_idx is not None:
            core_id, patch_idx = self.indices[patch_idx]

        if core_id is None:
            raise ValueError("Either patch_idx or core_idx must be specified")

        return self._bmode_finder.get_image(core_id)

    def filter_positions_by_mask(self, positions, mask, mask_name, threshold=0.5):
        positions_new = []
        for i in range(len(positions)):
            xmin_mm, ymin_mm, xmax_mm, ymax_mm = positions[i]["position"]
            xmin_px, ymin_px = np.round(
                convert_physical_coordinate_to_pixel_coordinate(
                    (xmin_mm, ymin_mm), (28, 46.06), mask.shape
                )
            ).astype(int)
            xmax_px, ymax_px = np.round(
                convert_physical_coordinate_to_pixel_coordinate(
                    (xmax_mm, ymax_mm), (28, 46.06), mask.shape
                )
            ).astype(int)

            mask_patch = mask[xmin_px:xmax_px, ymin_px:ymax_px]
            intersection = np.sum(mask_patch) / mask_patch.size

            logging.debug(
                f"Position, xmin_mm, ymin_mm, xmax_mm, ymax_mm: {xmin_mm}, {ymin_mm}, {xmax_mm}, {ymax_mm}"
            )
            logging.debug(f"Intersection: {intersection}")

            if intersection > threshold:
                positions_new.append(
                    {**positions[i], f"{mask_name}_mask_intersection": intersection}
                )
        return positions_new

    def show_patch(self, idx):
        core_id, patch_idx = self.indices[idx]
        core_metadata = self.get_metadata_for_id(core_id)
        item = self[idx]
        patch = item["patch"]
        xmin, ymin, xmax, ymax = item["position"]

        # bmode = self._data_array[self._core_id_to_idx[core_id]]
        bmode = self._bmode_finder.get_image(core_id)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(bmode, extent=(0, 46.06, 28, 0))

        if self.patch_options.prostate_mask_threshold is not None:
            prostate_mask = lookup_prostate_mask(core_id)
            ax[0].imshow(prostate_mask, alpha=0.5, extent=(0, 46.06, 28, 0))
        ax[0].plot(
            [ymin, ymax, ymax, ymin, ymin], [xmin, xmin, xmax, xmax, xmin], color="red"
        )
        ax[1].imshow(patch[..., 0] if patch.ndim == 3 else patch)
        fig.suptitle(f"{core_id} {core_metadata['grade']}")

    def get_metadata_for_id(self, core_id):
        out = dict(self.metadata[self.metadata["core_id"] == core_id].iloc[0])
        out["core_specifier"] = out["core_id"]
        return out

    @property
    def labels(self):
        for i in range(len(self)):
            core_id, _ = self.indices[i]
            core_metadata = self.get_metadata_for_id(core_id)
            yield int(core_metadata["grade"] != "Benign")

    def __repr__(self):
        return f"ExactDataset(patches={len(self)}, cores={len(self.core_ids)})"

    def position_wise_indices(self, switch_position_every_n_samples=None): 
        """Creates a sampler that samples patches position by position"""

        unique_coordinates = {}
        """This dictionary's keys will be the unique coordinates of the patches, and the values will be a list of tuples (core_id, idx) where idx is the index of the patch in the core"""
        for core_id in self.core_ids:
            for idx, position in enumerate(self.positions[core_id]):
                unique_coordinates.setdefault(position['position'], []).append((core_id, idx))

        # we need to go backwards from core_id, idx to index
        index_to_core_id_and_idx = {multi_index: i for i, multi_index in enumerate(self.indices)}

        # we should shuffle the positions
        unique_coordinates_keys = list(unique_coordinates.keys())
        np.random.shuffle(unique_coordinates_keys)

        # for each position, we need to shuffle the patches
        for key in unique_coordinates_keys:
            np.random.shuffle(unique_coordinates[key])

        # now we can iterate: 

        while len(unique_coordinates_keys) > 0: 
            num_yielded_for_position = 0
            key = unique_coordinates_keys[0]
            while len(unique_coordinates[key]) > 0: 
                core_id, idx = unique_coordinates[key].pop()
                yield index_to_core_id_and_idx[(core_id, idx)]
                num_yielded_for_position += 1
                if switch_position_every_n_samples is not None and num_yielded_for_position >= switch_position_every_n_samples: 
                    break
            if len(unique_coordinates[key]) == 0: 
                unique_coordinates_keys.pop(0)
            else: 
                np.random.shuffle(unique_coordinates_keys)

    def patient_wise_batch_indices(self): 
        core_ids = sorted(self.core_ids)
        idx_inverse = {index: i for i, index in enumerate(self.indices)}
        all_indices_for_core_id = {
            core_id: [v for k, v in idx_inverse.items() if k[0]==core_id] for core_id in core_ids
        }
        patient_ids = self.metadata.query('core_id in @core_ids').patient_id.unique().tolist()

        patient_id_to_core_ids = {}
        for patient_id in patient_ids: 
            for core_id in core_ids: 
                if core_id.startswith(patient_id): 
                    patient_id_to_core_ids.setdefault(patient_id, []).append(core_id)

        for patient_id in patient_ids:
            indices = []
            for core_id in patient_id_to_core_ids[patient_id]:
                indices.extend(all_indices_for_core_id[core_id])
            yield indices
            

class ShiftedPatchPairsDataset:
    def __init__(
        self,
        core_ids,
        mode: tp.Literal[
            "bmode", "rf", "split_channel_image", "envelope", "theta"
        ] = "bmode",
        transform=None,
        target_transform=None,
        patch_options: PatchOptions = PatchOptions(),
    ):
        self._dataset = PatchesDatasetV3(
            core_ids=core_ids,
            mode=mode,
            transform=transform,
            target_transform=target_transform,
            patch_options=patch_options,
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item1 = self._dataset[idx]
        item2 = self._dataset[idx]
        position1 = item1["position"]
        position2 = item2.pop("position")

        patch1 = item1["patch"]
        patch2 = item2.pop("patch")
        item2["patch"] = patch1, patch2
        item2["position"] = position1, position2
        return item2

    def position_wise_indices(self, switch_position_every_n_samples=None):
        return self._dataset.position_wise_indices(switch_position_every_n_samples)


class CoresDataset(torch.utils.data.Dataset):
    """Similar to patches dataset but will return all the patches for a core
    concatenated together"""

    def __init__(
        self,
        core_ids,
        mode: tp.Literal["rf", "bmode_undersampled"] = "bmode_undersampled",
        patch_options: PatchOptions = PatchOptions(),
        patch_transform=default_patch_transform,
    ):
        self.core_ids = core_ids
        self.mode = mode
        self.patch_options = patch_options
        self.patch_transform = patch_transform
        self._image_finder = ImageFinderV2(mode)

        core_ids = []
        failed_cores = []
        for core_id in self.core_ids:
            try:
                self._image_finder.get_image(core_id)
                core_ids.append(core_id)
            except:
                failed_cores.append(core_id)
        logging.info(f"Failed cores: {failed_cores}")
        self.core_ids = core_ids

        base_positions = list(compute_base_positions((28, 46.06), patch_options))

        base_positions = list(
            compute_mask_intersections(
                base_positions,
                needle_mask(),
                "needle",
                (28, 46.06),
                threshold=patch_options.needle_mask_threshold,
            )
        )

        position_data_for_cores = []
        failed_cores = []
        for core in tqdm(core_ids):
            if patch_options.prostate_mask_threshold > 0:
                try:
                    prostate_mask = lookup_prostate_mask(core)
                    position_data = (
                        list(
                            compute_mask_intersections(
                                base_positions,
                                prostate_mask,
                                "prostate",
                                (28, 46.06),
                                threshold=patch_options.prostate_mask_threshold,
                            )
                        )
                    )
                except:
                    # if there's no prostate mask we just ignore it
                    position_data = base_positions
            else:
                position_data = base_positions
            if len(position_data) == 0:
                failed_cores.append(core)
            else:
                position_data_for_cores.append(position_data)
        self.core_ids = [
            core_id for core_id in self.core_ids if core_id not in failed_cores
        ]
        print(f"Failed cores: {failed_cores}")
        self.position_data_for_cores = position_data_for_cores

    def __len__(self):
        return len(self.core_ids)

    def __getitem__(self, idx):
        core_id = self.core_ids[idx]
        core_metadata = get_metadata_for_id(core_id)
        positions = copy.deepcopy(self.position_data_for_cores[idx]) # we need to deepcopy because we modify the positions in place
        image = self._image_finder.get_image(core_id)

        patches = []
        for i in range(len(positions)):
            position_datum = positions[i]
            xmin_mm, ymin_mm, xmax_mm, ymax_mm = position_datum["position"]
            if self.patch_options.shift_delta_mm > 0:
                xmin_mm, ymin_mm, xmax_mm, ymax_mm = shift_patch_position(
                    (xmin_mm, ymin_mm, xmax_mm, ymax_mm),
                    self.patch_options.shift_delta_mm,
                )
                position_datum["position"] = (
                    xmin_mm,
                    ymin_mm,
                    xmax_mm,
                    ymax_mm,
                )
            xmin_px, ymin_px = convert_physical_coordinate_to_pixel_coordinate(
                (xmin_mm, ymin_mm), (28, 46.06), image.shape[:2]
            )
            xmax_px, ymax_px = convert_physical_coordinate_to_pixel_coordinate(
                (xmax_mm, ymax_mm), (28, 46.06), image.shape[:2]
            )
            patch = image[xmin_px:xmax_px, ymin_px:ymax_px]
            if self.patch_transform is not None:
                patch = self.patch_transform(patch)

            patches.append(patch)

        patches = np.stack(patches)

        position_output = {}
        for key in positions[0].keys():
            position_output[key] = np.array(
                [position_datum[key] for position_datum in positions]
            )

        sample = {
            "patches": patches,
            **position_output,
            **core_metadata,
            "label": torch.tensor(core_metadata["grade"] != "Benign", dtype=torch.int64),
        }
        return sample


class RFAndBModeDataset(torch.utils.data.Dataset):
    """
    A dataset that returns pairs of RF and B-mode patches,
    potentially to be used for a Siamese network.
    """

    def __init__(
        self,
        core_ids,
        transform,
        target_transform,
        patch_options: PatchOptions = PatchOptions(),
    ):
        self.rf_dataset = PatchesDatasetV3(
            core_ids=core_ids,
            mode="rf",
            transform=transform,
            target_transform=target_transform,
            patch_options=patch_options,
        )

    def __len__(self):
        return len(self.rf_dataset)

    def __getitem__(self, idx):
        item1 = self.rf_dataset[idx]
        rf_patch = item1.pop("patch")
        position = item1["position"]
        xmin, ymin, xmax, ymax = position

        core_id = item1["core_id"]
        bmode_image = UndersampledBModeFinder.get_image(core_id)
        xmin_px, ymin_px = convert_physical_coordinate_to_pixel_coordinate(
            (xmin, ymin), (28, 46.06), bmode_image.shape[:2]
        )
        xmax_px, ymax_px = convert_physical_coordinate_to_pixel_coordinate(
            (xmax, ymax), (28, 46.06), bmode_image.shape[:2]
        )
        bmode_patch = bmode_image[xmin_px:xmax_px, ymin_px:ymax_px]
        if self.rf_dataset.transform is not None:
            bmode_patch = self.rf_dataset.transform(bmode_patch)
        item1["bmode_patch"] = bmode_patch
        item1["rf_patch"] = rf_patch

        return item1

    def show_patch(self, idx):
        self.bmode_dataset.show_patch(idx)
        plt.figure()
        self.rf_dataset.show_patch(idx)


@dataclass
class DatasetConfig(SplitsConfig):
    """Configuration for the dataset factory."""

    normalization: str = "instance"
    augmentations_mode: tp.Literal["none", "cv_augs"] = "none"
    imaging_mode: tp.Literal[
        "rf", "bmode", "envelope", "theta", "split_channel_image", "bmode_undersampled"
    ] = "rf"
    input_size: tp.Tuple[int, int] | None = (447, 56)
    patch_options: PatchOptions = field(default_factory=PatchOptions)


BMODE_PATCH_MEAN = 6.03748642971247
BMODE_PATCH_STD = 0.8387530037126129

RF_PATCHES_MEAN = 0.49899267381964224
RF_PATCHES_STD = 924.0411699457405


@staticmethod
def label_transform(label):
    return torch.tensor(label, dtype=torch.int64)


def create_transform(args: DatasetConfig, mode):
    def transform(patch):
        if args.normalization == "instance":
            patch = (patch - patch.mean()) / patch.std()

        elif args.normalization == "global":
            if args.imaging_mode == "bmode":
                patch = (patch - BMODE_PATCH_MEAN) / BMODE_PATCH_STD
            else:
                patch = (patch - RF_PATCHES_MEAN) / RF_PATCHES_STD

        if args.input_size is not None:
            import cv2

            patch = cv2.resize(patch, (args.input_size[1], args.input_size[0]))

        patch = torch.tensor(patch, dtype=torch.float32)

        if args.imaging_mode == "split_channel_image":
            patch = patch.permute(2, 0, 1)

        else:
            patch = patch[None, :, :]

        # augmentations
        from torchvision import transforms as T

        if mode == "train" and args.augmentations_mode == "cv_augs":
            patch = T.RandomResizedCrop(args.input_size, scale=(0.8, 1.0))(patch)
            patch = T.RandomHorizontalFlip()(patch)
            patch = T.RandomVerticalFlip()(patch)
            patch = T.RandomAffine(0, translate=(0.1, 0.1))(patch)

        return patch

    return transform


def create_dataset(args: DatasetConfig):
    train_cores, val_cores, test_cores = generate_splits(args)

    train_transform = create_transform(args, mode="train")
    eval_transform = create_transform(args, mode="eval")

    train_dataset = PatchesDatasetV3(
        core_ids=train_cores,
        mode=args.imaging_mode,
        transform=train_transform,
        target_transform=label_transform,
        patch_options=args.patch_options,
    )

    val_dataset = PatchesDatasetV3(
        core_ids=val_cores,
        mode=args.imaging_mode,
        transform=eval_transform,
        target_transform=label_transform,
        patch_options=args.patch_options,
    )

    test_dataset = PatchesDatasetV3(
        core_ids=test_cores,
        mode=args.imaging_mode,
        transform=eval_transform,
        target_transform=label_transform,
        patch_options=args.patch_options,
    )

    return train_dataset, val_dataset, test_dataset
