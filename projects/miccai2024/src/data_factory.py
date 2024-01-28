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
    def __init__(self, augment="translate", image_size=1024, dataset_name="nct"):
        self.augment = augment
        self.image_size = image_size
        self.dataset_name = dataset_name

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
        out["all_cores_benign"] = torch.tensor(item["all_cores_benign"]).bool()
        out["dataset_name"] = self.dataset_name

        return out


class BModeDataFactoryV1(DataFactory):
    def __init__(
        self,
        fold: int = 0,
        n_folds: int = 5,
        test_center: str | None = None,
        undersample_benign_ratio: float = 3.0,
        min_involvement_train: float = 40,
        remove_benign_cores_from_positive_patients: bool = True,
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
            exclude_benign_cores_from_positive_patients=remove_benign_cores_from_positive_patients,
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


class AlignedFilesSegmentationDataFactory(DataFactory):
    def __init__(self, batch_size: int = 1, image_size: int = 1024, augmentations="none"):
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

        bmode = item['image']
        bmode = np.flip(bmode, axis=0).copy()
        bmode = torch.from_numpy(bmode) / 255.0
        bmode = bmode.unsqueeze(0)
        bmode = T.Resize((1024, 1024), antialias=True)(bmode)
        bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
        bmode = bmode.repeat(3, 1, 1)
        bmode = Image(bmode)

        prostate_mask = item['mask']
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

        out["label"] = torch.tensor(-1).long()
        out["involvement"] = torch.tensor(0).float()
        out["psa"] = torch.tensor(-1).float()
        out["age"] = torch.tensor(-1).float()
        out["family_history"] = torch.tensor(False).bool()
        out["center"] = 'UCLA'
        out["loc"] = 'UNKNOWN'
        out["all_cores_benign"] = torch.tensor(False).bool()
        out["dataset_name"] = "aligned_files"

        return out

    def train_transform(self, item):
        return self.transform(item, augmentations=self.augmentations)
    
    def val_transform(self, item):
        return self.transform(item, augmentations='none')

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


def dataloaders(
    fold=0,
    num_folds=5,
    batch_size=1,
    train_dataset_names: list[str] = ["nct", "aligned_files"],
    test_dataset_names: list[str] = ["nct", "aligned_files"],
):
    from medAI.datasets import (
        AlignedFilesDataset,
        ExactNCT2013BmodeImagesWithManualProstateSegmentation,
        CohortSelectionOptions,
    )
    from torchvision.transforms import v2 as T
    from torchvision.tv_tensors import Mask, Image
    from torchvision.transforms import InterpolationMode

    class AlignedFilesTransform:
        """Transforms for the aligned files dataset"""

        def __init__(self, augment=False):
            self.augment = augment

        def __call__(self, item):
            bmode = item["image"]
            bmode = T.ToTensor()(bmode)
            bmode = T.Resize((1024, 1024), antialias=True)(bmode)
            bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
            bmode = bmode.repeat(3, 1, 1)
            bmode = Image(bmode)

            mask = item["mask"]
            mask = mask.astype("uint8")
            mask = T.ToTensor()(mask).float()
            mask = T.Resize(
                (1024, 1024), antialias=False, interpolation=InterpolationMode.NEAREST
            )(mask)
            mask = Mask(mask)

            if self.augment:
                augmentation = T.Compose(
                    [
                        T.RandomApply([T.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
                        T.RandomApply(
                            [T.RandomResizedCrop(1024, scale=(0.8, 1.0))], p=0.5
                        ),
                    ]
                )
                bmode, mask = augmentation(bmode, Mask(mask))

            return bmode, mask

    class NCTTransform:
        def __init__(self, augment=False):
            self.augment = augment

        def __call__(self, item):
            bmode = item["bmode"]
            bmode = np.flip(bmode, axis=0).copy()
            bmode = T.ToTensor()(bmode)
            bmode = T.Resize((1024, 1024), antialias=True)(bmode)
            bmode = (bmode - bmode.min()) / (bmode.max() - bmode.min())
            bmode = bmode.repeat(3, 1, 1)
            bmode = Image(bmode)
            mask = item["prostate_mask"]
            mask = np.flip(mask, axis=0).copy()
            mask = mask.astype("uint8")
            mask = T.ToTensor()(mask).float() * 255
            mask = T.Resize(
                (1024, 1024), antialias=False, interpolation=InterpolationMode.NEAREST
            )(mask)
            mask = Mask(mask)
            if self.augment:
                augmentation = T.Compose(
                    [
                        T.RandomApply([T.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
                        T.RandomApply(
                            [T.RandomResizedCrop(1024, scale=(0.8, 1.0))], p=0.5
                        ),
                    ]
                )
                bmode, mask = augmentation(bmode, Mask(mask))

            return bmode, mask

    logging.info("Setting up datasets")
    train_datasets = []
    if "aligned_files" in train_dataset_names:
        train_datasets.append(
            AlignedFilesDataset(
                split="train",
                transform=AlignedFilesTransform(augment=True),
            )
        )
    if "nct" in train_dataset_names:
        train_datasets.append(
            ExactNCT2013BmodeImagesWithManualProstateSegmentation(
                split="train",
                transform=NCTTransform(augment=True),
                cohort_selection_options=CohortSelectionOptions(fold=fold, n_folds=num_folds),
            )
        )
    train_ds = torch.utils.data.ConcatDataset(train_datasets)

    test_datasets = {}
    if "aligned_files" in test_dataset_names:
        test_datasets["aligned_files"] = AlignedFilesDataset(
            split="test",
            transform=AlignedFilesTransform(augment=False),
        )
    if "nct" in test_dataset_names:
        test_datasets["nct"] = ExactNCT2013BmodeImagesWithManualProstateSegmentation(
            split="test",
            transform=NCTTransform(augment=False),
            cohort_selection_options=CohortSelectionOptions(fold=fold, n_folds=num_folds),
        )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loaders = {
        name: DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=4
        )
        for name, test_ds in test_datasets.items()
    }
    return train_loader, test_loaders