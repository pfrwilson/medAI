from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask
from .transform import RandomTranslation
from dataclasses import dataclass
import typing as tp 
from pydantic import BaseModel

from medAI.datasets.nct2013.data_access import data_accessor

table = data_accessor.get_metadata_table()
psa_min = table["psa"].min()
psa_max = table["psa"].max()
psa_avg = table["psa"].mean()
age_min = table["age"].min()
age_max = table["age"].max()
age_avg = table["age"].mean()
approx_psa_density_min = table["approx_psa_density"].min()
approx_psa_density_max = table["approx_psa_density"].max()
approx_psa_density_avg = table["approx_psa_density"].mean()


CORE_LOCATION_TO_IDX = {
    "LML": 0,
    "RBL": 1,
    "LMM": 2,
    "RMM": 2,
    "LBL": 1,
    "LAM": 3,
    "RAM": 3,
    "RML": 0,
    "LBM": 4,
    "RAL": 5,
    "RBM": 4,
    "LAL": 5,
}


class DataFactory(ABC):
    @abstractmethod
    def train_loader(self):
        ...

    @abstractmethod
    def val_loader(self):
        ...

    @abstractmethod
    def test_loader(self):
        ...

    def show(self):
        """Show something interesting and maybe log it to wandb."""
        ...

 
class TransformV1:
    def __init__(
        self,
        augment="translate",
        image_size=1024,
        mask_size=256,
        dataset_name="nct",
        labeled=True,
    ):
        self.augment = augment
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.labeled = labeled
        self.mask_size = mask_size

    def __call__(self, item):
        out = item.copy()
        bmode = item["bmode"]
        bmode = torch.from_numpy(bmode.copy()).float()
        bmode = bmode.unsqueeze(0)
        bmode = T.Resize((self.image_size, self.image_size), antialias=True)(bmode)
        bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
        bmode = bmode.repeat(3, 1, 1)
        bmode = Image(bmode)
        if not self.labeled:
            return {"bmode": bmode}

        needle_mask = item["needle_mask"]
        needle_mask = needle_mask = torch.from_numpy(needle_mask.copy()).float()
        needle_mask = needle_mask.unsqueeze(0)
        needle_mask = T.Resize(
            (self.image_size, self.image_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(needle_mask)
        needle_mask = Mask(needle_mask)

        prostate_mask = item["prostate_mask"]
        prostate_mask = prostate_mask = torch.from_numpy(prostate_mask.copy()).float()
        prostate_mask = prostate_mask.unsqueeze(0)
        prostate_mask = T.Resize(
            (self.image_size, self.image_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(prostate_mask)
        prostate_mask = Mask(prostate_mask)

        if self.augment == "translate":
            bmode, needle_mask, prostate_mask = T.RandomAffine(
                degrees=0, translate=(0.2, 0.2)
            )(bmode, needle_mask, prostate_mask)
        elif self.augment == "translate_crop":
            bmode, needle_mask, prostate_mask = T.RandomAffine(
                degrees=0, translate=(0.2, 0.2)
            )(bmode, needle_mask, prostate_mask)
            bmode, needle_mask, prostate_mask = T.RandomResizedCrop(
                size=(self.image_size, self.image_size), scale=(0.8, 1.0)
            )(bmode, needle_mask, prostate_mask)
        elif self.augment == "resized_crop":
            bmode, needle_mask, prostate_mask = T.RandomResizedCrop(
                size=(self.image_size, self.image_size), scale=(0.8, 1.0)
            )(bmode, needle_mask, prostate_mask)
        elif self.augment == "crop_random_gamma":
            random_gamma = (0.5, 1.5)
            crop_scale_1 = (0.8, 1.0)
            if np.random.rand() < 0.5:
                from torchvision.transforms.functional import adjust_gamma

                gamma = np.random.uniform(*random_gamma)
                bmode = adjust_gamma(bmode, gamma, gain=gamma)

            if crop_scale_1 is not None:
                from torchvision.transforms.v2 import RandomResizedCrop

                crop1 = RandomResizedCrop((1024, 1024), scale=crop_scale_1)
                bmode, needle_mask, prostate_mask = crop1(
                    bmode, needle_mask, prostate_mask
                )
        elif self.augment == "strong_crop":
            crop = T.RandomResizedCrop((1024, 1024), scale=(0.5, 1.0), antialias=True)
            bmode, needle_mask, prostate_mask = crop(bmode, needle_mask, prostate_mask)

        # interpolate the masks to the mask size
        needle_mask = T.Resize(
            (self.mask_size, self.mask_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(needle_mask)
        prostate_mask = T.Resize(
            (self.mask_size, self.mask_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(prostate_mask)

        out["bmode"] = bmode
        out["needle_mask"] = needle_mask
        out["prostate_mask"] = prostate_mask

        out["label"] = torch.tensor(item["grade"] != "Benign").long()
        pct_cancer = item["pct_cancer"]
        if np.isnan(pct_cancer):
            pct_cancer = 0
        out["involvement"] = torch.tensor(pct_cancer / 100).float()

        psa = item["psa"]
        if np.isnan(psa):
            psa = psa_avg
        psa = (psa - psa_min) / (psa_max - psa_min)
        out["psa"] = torch.tensor([psa]).float()

        age = item["age"]
        if np.isnan(age):
            age = age_avg
        age = (age - age_min) / (age_max - age_min)
        out["age"] = torch.tensor([age]).float()

        approx_psa_density = item["approx_psa_density"]
        if np.isnan(approx_psa_density):
            approx_psa_density = approx_psa_density_avg
        approx_psa_density = (approx_psa_density - approx_psa_density_min) / (
            approx_psa_density_max - approx_psa_density_min
        )
        out["approx_psa_density"] = torch.tensor([approx_psa_density]).float()

        if item["family_history"] is True:
            out["family_history"] = torch.tensor(1).long()
        elif item["family_history"] is False:
            out["family_history"] = torch.tensor(0).long()
        elif np.isnan(item["family_history"]):
            out["family_history"] = torch.tensor(2).long()

        out["center"] = item["center"]
        loc = item["loc"]
        out["loc"] = torch.tensor(CORE_LOCATION_TO_IDX[loc]).long()
        out["all_cores_benign"] = torch.tensor(item["all_cores_benign"]).bool()
        out["dataset_name"] = self.dataset_name
        out["primary_grade"] = item["primary_grade"]
        out["secondary_grade"] = item["secondary_grade"]
        out["grade"] = item["grade"]
        out["core_id"] = item["core_id"]

        return out


class TransformV2:
    def __init__(
        self,
        augment="translate",
        image_size=1024,
        mask_size=256,
        dataset_name="nct",
        labeled=True,
    ):
        self.augment = augment
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.labeled = labeled
        self.mask_size = mask_size

    def __call__(self, item):
        out = item.copy()
        bmode = item["bmode"]
        bmode = torch.from_numpy(bmode.copy()).float()
        bmode = bmode.unsqueeze(0)
        bmode = T.Resize((self.image_size, self.image_size), antialias=True)(bmode)
        bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
        bmode = bmode.repeat(3, 1, 1)
        bmode = Image(bmode)
        if not self.labeled:
            return {"bmode": bmode}

        needle_mask = item["needle_mask"]
        needle_mask = needle_mask = torch.from_numpy(needle_mask.copy()).float()
        needle_mask = needle_mask.unsqueeze(0)
        needle_mask = T.Resize(
            (self.image_size, self.image_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(needle_mask)
        needle_mask = Mask(needle_mask)

        prostate_mask = item["prostate_mask"]
        prostate_mask = prostate_mask = torch.from_numpy(prostate_mask.copy()).float()
        prostate_mask = prostate_mask.unsqueeze(0)
        prostate_mask = T.Resize(
            (self.image_size, self.image_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(prostate_mask)
        prostate_mask = Mask(prostate_mask)

        if item.get("rf") is not None:
            rf = item["rf"]
            rf = torch.from_numpy(rf.copy()).float()
            rf = rf.unsqueeze(0)
            if rf.shape != (2504, 512):
                rf = T.Resize((2504, 512), antialias=True)(rf)
            rf = rf.repeat(3, 1, 1)

            if self.augment == "translate":
                bmode, rf, needle_mask, prostate_mask = RandomTranslation(
                    translation=(0.2, 0.2)
                )(bmode, rf, needle_mask, prostate_mask)

        else:
            bmode, needle_mask, prostate_mask = RandomTranslation(
                translation=(0.2, 0.2)
            )(bmode, needle_mask, prostate_mask)

        # interpolate the masks to the mask size
        needle_mask = T.Resize(
            (self.mask_size, self.mask_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(needle_mask)
        prostate_mask = T.Resize(
            (self.mask_size, self.mask_size),
            antialias=False,
            interpolation=InterpolationMode.NEAREST,
        )(prostate_mask)

        out["bmode"] = bmode
        out["needle_mask"] = needle_mask
        out["prostate_mask"] = prostate_mask

        if item.get("rf") is not None:
            out["rf"] = rf

        out["label"] = torch.tensor(item["grade"] != "Benign").long()
        pct_cancer = item["pct_cancer"]
        if np.isnan(pct_cancer):
            pct_cancer = 0
        out["involvement"] = torch.tensor(pct_cancer / 100).float()

        psa = item["psa"]
        if np.isnan(psa):
            psa = psa_avg
        psa = (psa - psa_min) / (psa_max - psa_min)
        out["psa"] = torch.tensor([psa]).float()

        age = item["age"]
        if np.isnan(age):
            age = age_avg
        age = (age - age_min) / (age_max - age_min)
        out["age"] = torch.tensor([age]).float()

        approx_psa_density = item["approx_psa_density"]
        if np.isnan(approx_psa_density):
            approx_psa_density = approx_psa_density_avg
        approx_psa_density = (approx_psa_density - approx_psa_density_min) / (
            approx_psa_density_max - approx_psa_density_min
        )
        out["approx_psa_density"] = torch.tensor([approx_psa_density]).float()

        if item["family_history"] is True:
            out["family_history"] = torch.tensor(1).long()
        elif item["family_history"] is False:
            out["family_history"] = torch.tensor(0).long()
        elif np.isnan(item["family_history"]):
            out["family_history"] = torch.tensor(2).long()

        out["center"] = item["center"]
        loc = item["loc"]
        out["loc"] = torch.tensor(CORE_LOCATION_TO_IDX[loc]).long()
        out["all_cores_benign"] = torch.tensor(item["all_cores_benign"]).bool()
        out["dataset_name"] = self.dataset_name
        out["primary_grade"] = item["primary_grade"]
        out["secondary_grade"] = item["secondary_grade"]
        out["grade"] = item["grade"]
        out["core_id"] = item["core_id"]

        return out


class BModeDataFactoryV1Config(BaseModel):
    """
    Args:
        fold (int): The fold to use. If not specified, uses leave-one-center-out cross-validation.
        n_folds (int): The number of folds to use for cross-validation.
        test_center (str): If not None, uses leave-one-center-out cross-validation with the specified center as test. If None, uses k-fold cross-validation.
        undersample_benign_ratio (float): If not None, undersamples benign cores with the specified ratio.
        min_involvement_train (float): The minimum involvement threshold to use for training.
        remove_benign_cores_from_positive_patients (bool): If True, removes benign cores from positive patients (training only).
        batch_size (int): The batch size to use for training.
        image_size (int): The size to use for the images.
        mask_size (int): The size to use for the masks.
        augmentations (str): The augmentations to use for training.
        labeled (bool): If True, uses labeled data.
        limit_train_data (float): If not None, limits the amount of training data to the specified ratio.
        train_subset_seed (int): The seed to use for the train subset split.
        val_seed (int): The seed to use for validation split.
        rf_as_bmode (bool): If True, uses the radiofrequency data as BMode data.
        include_rf (bool): If True, includes the radiofrequency data in the dataset.
    """

    fold: int | None = None
    n_folds: int | None = None
    test_center: str | None  = 'UVA'
    undersample_benign_ratio: float | None = None
    min_involvement_train: float | None = 40
    remove_benign_cores_from_positive_patients: bool = True
    batch_size: int = 1
    image_size: int = 1024
    mask_size: int = 256
    augmentations: tp.Literal["none", "translate"] = "translate" 
    labeled: bool = True 
    limit_train_data: float | None = None
    train_subset_seed: int = 0
    val_seed: int = 0
    rf_as_bmode: bool = False 
    include_rf: bool = False 


class BModeDataFactoryV1(DataFactory):
    def __init__(self, config: BModeDataFactoryV1Config):
        self.cfg = config

        from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
        from medAI.datasets.nct2013.cohort_selection import select_cohort

        train_cores, val_cores, test_cores = select_cohort(
            fold=self.cfg.fold,
            n_folds=self.cfg.n_folds,
            test_center=self.cfg.test_center,
            undersample_benign_ratio=self.cfg.undersample_benign_ratio,
            involvement_threshold_pct=self.cfg.min_involvement_train,
            exclude_benign_cores_from_positive_patients=self.cfg.remove_benign_cores_from_positive_patients,
            splits_file="/ssd005/projects/exactvu_pca/nct2013/patient_splits.csv",
            val_seed=self.cfg.val_seed,
        )

        if self.cfg.limit_train_data is not None:
            cores = train_cores
            center = [core.split("-")[0] for core in cores]
            from sklearn.model_selection import StratifiedShuffleSplit

            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1 - self.cfg.limit_train_data,
                random_state=self.cfg.train_subset_seed,
            )
            for train_index, _ in sss.split(cores, center):
                train_cores = [cores[i] for i in train_index]

        self.train_transform = TransformV2(
            augment=self.cfg.augmentations,
            image_size=self.cfg.image_size,
            labeled=self.cfg.labeled,
            mask_size=self.cfg.mask_size,
        )
        self.val_transform = TransformV2(
            augment="none",
            image_size=self.cfg.image_size,
            labeled=self.cfg.labeled,
            mask_size=self.cfg.mask_size,
        )

        self.train_dataset = BModeDatasetV1(
            train_cores,
            self.train_transform,
            rf_as_bmode=self.cfg.rf_as_bmode,
            include_rf=self.cfg.include_rf,
        )
        self.val_dataset = BModeDatasetV1(
            val_cores,
            self.val_transform,
            rf_as_bmode=self.cfg.rf_as_bmode,
            include_rf=self.cfg.include_rf,
        )
        self.test_dataset = BModeDatasetV1(
            test_cores,
            self.val_transform,
            rf_as_bmode=self.cfg.rf_as_bmode,
            include_rf=self.cfg.include_rf,
        )

        self.batch_size = self.cfg.batch_size
        self.labeled = self.cfg.labeled

    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )


class AlignedFilesSegmentationDataFactory(DataFactory):
    def __init__(
        self, batch_size: int = 1, image_size: int = 1024, augmentations="none"
    ):
        from medAI.datasets import AlignedFilesDataset

        self.train_dataset = AlignedFilesDataset(
            split="train", transform=self.train_transform
        )
        self.val_dataset = AlignedFilesDataset(
            split="test", transform=self.val_transform
        )
        self.test_dataset = AlignedFilesDataset(
            split="test", transform=self.val_transform
        )

        self.batch_size = batch_size
        self.augmentations = augmentations
        self.image_size = image_size

    def transform(self, item, augmentations="none"):
        out = {}

        bmode = item["image"]
        bmode = np.flip(bmode, axis=0).copy()
        bmode = torch.from_numpy(bmode) / 255.0
        bmode = bmode.unsqueeze(0)
        bmode = T.Resize((1024, 1024), antialias=True)(bmode)
        bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
        bmode = bmode.repeat(3, 1, 1)
        bmode = Image(bmode)

        prostate_mask = item["mask"]
        prostate_mask = np.flip(prostate_mask, axis=0).copy()
        prostate_mask = torch.from_numpy(prostate_mask) / 255.0
        prostate_mask = prostate_mask.unsqueeze(0)
        prostate_mask = T.Resize((1024, 1024), antialias=True)(prostate_mask)
        prostate_mask = Mask(prostate_mask)

        needle_mask = torch.zeros_like(prostate_mask)

        if augmentations == "translate":
            bmode, needle_mask, prostate_mask = T.RandomAffine(
                degrees=0, translate=(0.2, 0.2)
            )(bmode, needle_mask, prostate_mask)
        elif augmentations == "translate_crop":
            bmode, needle_mask, prostate_mask = T.RandomAffine(
                degrees=0, translate=(0.2, 0.2)
            )(bmode, needle_mask, prostate_mask)
            bmode, needle_mask, prostate_mask = T.RandomResizedCrop(
                size=(self.image_size, self.image_size), scale=(0.8, 1.0)
            )(bmode, needle_mask, prostate_mask)
        elif augmentations == "resized_crop":
            bmode, needle_mask, prostate_mask = T.RandomResizedCrop(
                size=(self.image_size, self.image_size), scale=(0.8, 1.0)
            )(bmode, needle_mask, prostate_mask)

        out["bmode"] = bmode
        out["needle_mask"] = needle_mask
        out["prostate_mask"] = prostate_mask
        out["dataset_name"] = "aligned_files"

        return out

    def train_transform(self, item):
        return self.transform(item, augmentations=self.augmentations)

    def val_transform(self, item):
        return self.transform(item, augmentations="none")

    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )


class UASweepsDataFactory(DataFactory):
    """Data factory for UASweeps dataset, which are unlabeled BMode images"""

    def __init__(self, batch_size: int = 1, image_size: int = 1024):
        from medAI.datasets.ua_unlabeled import UAUnlabeledImages

        self.dataset = UAUnlabeledImages(
            root="/ssd005/projects/exactvu_pca/UA_extracted_data",
            transform=self.transform,
        )

        self.batch_size = batch_size
        self.image_size = image_size

    def transform(self, item):
        bmode = item["bmode"]
        bmode = np.flip(bmode, axis=0).copy()
        bmode = torch.from_numpy(bmode) / 255.0
        bmode = bmode.unsqueeze(0)
        bmode = T.Resize((self.image_size, self.image_size), antialias=True)(bmode)
        bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
        bmode = bmode.repeat(3, 1, 1)
        bmode = Image(bmode)

        return {"bmode": bmode}

    def train_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_loader(self):
        return None

    def test_loader(self):
        return None

