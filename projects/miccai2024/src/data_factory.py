from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask

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

    @classmethod
    def add_argparse_args(cls, parser):
        import inspect

        sig = inspect.signature(cls.__init__)
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            default = param.default if param.default != inspect._empty else None
            name = param.name
            type = param.annotation if param.annotation != inspect._empty else str
            parser.add_argument(f"--{name}", type=type, default=default)

    @classmethod
    def from_argparse_args(cls, args):
        import inspect

        sig = inspect.signature(cls.__init__)
        kwargs = {k: v for k, v in vars(args).items() if k in sig.parameters}
        return cls(**kwargs)


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

        rf = item["rf"]
        rf = torch.from_numpy(rf.copy()).float()
        rf = rf.unsqueeze(0)
        if rf.shape != (2504, 512):
            rf = T.Resize((2504, 512), antialias=True)(rf)
        rf = rf.repeat(3, 1, 1)

        if self.augment == "translate":
            from .transform import RandomTranslation

            bmode, rf, needle_mask, prostate_mask = RandomTranslation(
                translation=(0.2, 0.2)
            )(bmode, rf, needle_mask, prostate_mask)

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


class BModeDataFactoryV1(DataFactory):
    def __init__(
        self,
        fold: int = 0,
        n_folds: int = 5,
        test_center: str = None,
        undersample_benign_ratio: float = 3.0,
        min_involvement_train: float = 40,
        remove_benign_cores_from_positive_patients: bool = True,
        batch_size: int = 1,
        image_size: int = 1024,
        mask_size: int = 256,
        augmentations: str = "none",
        labeled: bool = True,
        limit_train_data: float | None = None,
        train_subset_seed: int = 0,
        val_seed: int = 0,
        rf_as_bmode: bool = False,
    ):
        from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
        from medAI.datasets.nct2013.cohort_selection import select_cohort

        train_cores, val_cores, test_cores = select_cohort(
            fold=fold,
            n_folds=n_folds,
            test_center=test_center,
            undersample_benign_ratio=undersample_benign_ratio,
            involvement_threshold_pct=min_involvement_train,
            exclude_benign_cores_from_positive_patients=remove_benign_cores_from_positive_patients,
            splits_file="/ssd005/projects/exactvu_pca/nct2013/patient_splits.csv",
            val_seed=val_seed,
        )

        if limit_train_data is not None:
            cores = train_cores
            center = [core.split("-")[0] for core in cores]
            from sklearn.model_selection import StratifiedShuffleSplit

            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1 - limit_train_data,
                random_state=train_subset_seed,
            )
            for train_index, _ in sss.split(cores, center):
                train_cores = [cores[i] for i in train_index]

        self.train_transform = TransformV2(
            augment=augmentations,
            image_size=image_size,
            labeled=labeled,
            mask_size=mask_size,
        )
        self.val_transform = TransformV2(
            augment="none", image_size=image_size, labeled=labeled, mask_size=mask_size
        )

        self.train_dataset = BModeDatasetV1(
            train_cores, self.train_transform, include_rf=True, rf_as_bmode=rf_as_bmode
        )
        self.val_dataset = BModeDatasetV1(
            val_cores, self.val_transform, include_rf=True, rf_as_bmode=rf_as_bmode
        )
        self.test_dataset = BModeDatasetV1(
            test_cores, self.val_transform, include_rf=True, rf_as_bmode=rf_as_bmode
        )

        self.batch_size = batch_size
        self.labeled = labeled

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

    @staticmethod
    def add_args(parser):
        # fmt: off
        group = parser.add_argument_group("Data")
        group.add_argument("--fold", type=int, default=None, help="The fold to use. If not specified, uses leave-one-center-out cross-validation.")
        group.add_argument("--n_folds", type=int, default=None, help="The number of folds to use for cross-validation.")
        group.add_argument("--test_center", type=str, default=None, 
                            help="If not None, uses leave-one-center-out cross-validation with the specified center as test. If None, uses k-fold cross-validation.")
        group.add_argument("--val_seed", type=int, default=0, 
                        help="The seed to use for validation split.")            
        group.add_argument("--undersample_benign_ratio", type=lambda x: float(x) if not x.lower() == 'none' else None, default=None,
                        help="""If not None, undersamples benign cores with the specified ratio.""")
        group.add_argument("--min_involvement_train", type=float, default=0.0,
                        help="""The minimum involvement threshold to use for training.""")
        group.add_argument("--batch_size", type=int, default=1, help="The batch size to use for training.")
        group.add_argument("--augmentations", type=str, default="translate", help="The augmentations to use for training.")
        group.add_argument("--remove_benign_cores_from_positive_patients", action="store_true", help="If True, removes benign cores from positive patients (training only).")
        group.add_argument("--image_size", type=int, default=1024, help="The size to use for the images.")
        group.add_argument("--mask_size", type=int, default=256, help="The size to use for the masks.")
        # fmt: on

    @classmethod
    def from_args(cls, args, **kwargs):
        return cls(
            fold=args.fold,
            n_folds=args.n_folds,
            test_center=args.test_center,
            undersample_benign_ratio=args.undersample_benign_ratio,
            min_involvement_train=args.min_involvement_train,
            remove_benign_cores_from_positive_patients=args.remove_benign_cores_from_positive_patients,
            batch_size=args.batch_size,
            image_size=args.image_size,
            augmentations=args.augmentations,
            **kwargs,
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


def dataloaders(
    fold=0,
    num_folds=5,
    batch_size=1,
    train_dataset_names: list[str] = ["nct", "aligned_files"],
    test_dataset_names: list[str] = ["nct", "aligned_files"],
):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import v2 as T
    from torchvision.tv_tensors import Image, Mask

    from medAI.datasets import (
        AlignedFilesDataset,
        CohortSelectionOptions,
        ExactNCT2013BmodeImagesWithManualProstateSegmentation,
    )

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
                cohort_selection_options=CohortSelectionOptions(
                    fold=fold, n_folds=num_folds
                ),
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
            cohort_selection_options=CohortSelectionOptions(
                fold=fold, n_folds=num_folds
            ),
        )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loaders = {
        name: DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        for name, test_ds in test_datasets.items()
    }
    return train_loader, test_loaders
