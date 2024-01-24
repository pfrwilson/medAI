from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask
import numpy as np


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
    def __init__(self, augment="none", image_size=1024):
        self.augment = augment
        self.image_size = image_size

    def __call__(self, item):
        out = {}
        bmode = item["bmode"]
        bmode = torch.from_numpy(bmode.copy()).float()
        bmode = bmode.unsqueeze(0)
        bmode = T.Resize((self.image_size, self.image_size), antialias=True)(bmode)
        bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
        bmode = bmode.repeat(3, 1, 1)
        bmode = Image(bmode)

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

        if self.augment == "v1":
            bmode, needle_mask, prostate_mask = T.RandomAffine(
                degrees=0, translate=(0.2, 0.2)
            )(bmode, needle_mask, prostate_mask)
        elif self.augment == "v2":
            bmode, needle_mask, prostate_mask = T.RandomAffine(
                degrees=0, translate=(0.2, 0.2)
            )(bmode, needle_mask, prostate_mask)
            bmode, needle_mask, prostate_mask = T.RandomResizedCrop(
                size=(self.image_size, self.image_size), scale=(0.8, 1.0)
            )(bmode, needle_mask, prostate_mask)

        out["bmode"] = bmode
        out["needle_mask"] = needle_mask
        out["prostate_mask"] = prostate_mask

        out["label"] = torch.tensor(item["grade"] != "Benign").long()
        pct_cancer = item["pct_cancer"]
        if np.isnan(pct_cancer):
            pct_cancer = 0
        out["involvement"] = torch.tensor(pct_cancer / 100).float()
        out["psa"] = torch.tensor(item["psa"]).float()
        out["age"] = torch.tensor(item["age"]).float()
        out["family_history"] = torch.tensor(item["family_history"]).bool()
        out["center"] = item["center"]
        out["loc"] = item["loc"]

        return out


class BModeDataFactoryV1(DataFactory):
    def __init__(
        self,
        fold: int = 0,
        n_folds: int = 5,
        test_center: str | None = None,
        undersample_benign_ratio: float = 3.0,
        min_involvement_train: float = 40,
        batch_size: int = 1,
        image_size: int = 1024,
        augmentations: str = "none",
    ):
        from medAI.datasets.nct2013.cohort_selection import select_cohort
        from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1

        train_cores, val_cores, test_cores = select_cohort(
            fold=fold,
            n_folds=n_folds,
            test_center=test_center,
            undersample_benign_ratio=undersample_benign_ratio,
            involvement_threshold_pct=min_involvement_train,
            exclude_benign_cores_from_positive_patients=True,
            splits_file = "/ssd005/projects/exactvu_pca/nct2013/patient_splits.csv"
        )

        self.train_transform = TransformV1(augment=augmentations, image_size=image_size)
        self.val_transform = TransformV1(augment="none", image_size=image_size)

        self.train_dataset = BModeDatasetV1(train_cores, self.train_transform)
        self.val_dataset = BModeDatasetV1(val_cores, self.val_transform)
        self.test_dataset = BModeDatasetV1(test_cores, self.val_transform)

        self.batch_size = batch_size

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
