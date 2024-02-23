import logging
import os
from argparse import ArgumentParser

import numpy as np
import torch
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
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

import wandb

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
    group.add_argument("--full_prostate", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_weights_path", type=str, default="best_model.pth", help="Path to save the best model weights")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to save and load experiment state")

    # fmt: on
    return parser.parse_args()


def main(args):
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        state = torch.load(args.checkpoint_path)
    else:
        state = None

    wandb_run_id = state["wandb_run_id"] if state is not None else None
    run = wandb.init(
        project="miccai2024_ssl_debug", config=args, id=wandb_run_id, resume="allow"
    )
    wandb_run_id = run.id

    set_global_seed(args.seed)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    ssl_loader, train_loader, val_loader, test_loader = make_data_loaders(args)

    backbone = resnet10t_instance_norm()
    model = VICReg(backbone, proj_dims=[512, 512, 2048], features_dim=512).to(DEVICE)
    if state is not None:
        model.load_state_dict(state["model"])

    from medAI.utils.cosine_scheduler import cosine_scheduler

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])

    cosine_scheduler = cosine_scheduler(
        1e-4, 0, epochs=args.epochs, niter_per_ep=len(ssl_loader)
    )

    best_score = 0.0 if state is None else state["best_score"]
    start_epoch = 0 if state is None else state["epoch"]
    if state is not None:
        set_all_rng_states(state["rng_states"])

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch}")

        if args.checkpoint_path is not None:
            print("Saving checkpoint")
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
                "epoch": epoch,
                "rng_states": get_all_rng_states(),
                "wandb_run_id": wandb_run_id,
            }
            torch.save(state, args.checkpoint_path)

        print("Running SSL")
        model.train()
        for i, batch in enumerate(tqdm(ssl_loader)):
            # set lr
            iter = epoch * len(ssl_loader) + i
            lr = cosine_scheduler[iter]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            wandb.log({"lr": lr})

            optimizer.zero_grad()
            p1, p2 = batch
            p1, p2 = p1.to(DEVICE), p2.to(DEVICE)
            loss = model(p1, p2)
            wandb.log({"ssl_loss": loss.item()})
            loss.backward()
            optimizer.step()

        print("Running linear probing")
        model.eval()
        from medAI.utils.accumulators import DataFrameCollector

        accumulator = DataFrameCollector()

        X_train = []
        y_train = []
        for i, batch in enumerate(tqdm(train_loader)):
            patch = batch["patch"].to(DEVICE)
            y = torch.tensor(
                [0 if grade == "Benign" else 1 for grade in batch["grade"]],
                dtype=torch.long,
            )
            logging.debug(f"{patch.shape=}, {y.shape=}")
            with torch.no_grad():
                features = model.backbone(patch)
            logging.debug(f"{features.shape=}")
            X_train.append(features)
            y_train.append(y)
        X_train = torch.cat(X_train, dim=0)
        y_train = torch.cat(y_train)

        X_val = []
        for i, batch in enumerate(tqdm(val_loader)):
            patch = batch.pop("patch").to(DEVICE)
            accumulator(batch)
            with torch.no_grad():
                features = model.backbone(patch)
            X_val.append(features)

        X_val = torch.cat(X_val, dim=0)

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        clf = LogisticRegression(max_iter=10000)
        clf.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
        y_pred = clf.predict_proba(X_val.cpu().numpy())
        table = accumulator.compute()
        accumulator.reset()
        # insert predictions into table
        table.loc[:, "y_pred"] = y_pred[:, 1]

        y_pred_core = table.groupby("core_id")["y_pred"].mean()
        table["label"] = table.grade.apply(lambda g: 0 if g == "Benign" else 1)
        y_true_core = table.groupby("core_id")["label"].first()
        score = roc_auc_score(y_true_core, y_pred_core)

        high_involvement_table = table[
            (table.pct_cancer > 40) | (table.grade == "Benign")
        ]
        y_true_high_involvement = high_involvement_table.groupby("core_id")[
            "label"
        ].first()
        y_pred_high_involvement = high_involvement_table.groupby("core_id")[
            "y_pred"
        ].mean()
        score_high_involvement = roc_auc_score(
            y_true_high_involvement, y_pred_high_involvement
        )

        wandb.log(
            {"val_auc": score, "val_auc_high_involvement": score_high_involvement}
        )

        if score > best_score:
            best_score = score
            best_model_state = model.state_dict()
            torch.save(best_model_state, args.save_weights_path)


def resnet10t_instance_norm():
    from timm.models.resnet import resnet10t

    model = resnet10t(
        in_chans=3,
    )
    model.fc = nn.Identity()
    return nn.Sequential(nn.InstanceNorm2d(3), model)


def make_data_loaders(args):
    print(f"Preparing data loaders for test center {args.test_center}")

    train_patients, val_patients, test_patients = get_patient_splits_by_center(
        args.test_center, val_size=0.2, val_seed=args.val_seed
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

    print("SSL dataset...")
    ssl_dataset = BModePatchesDataset(
        ssl_train_core_ids,
        patch_size=(args.patch_size, args.patch_size),
        stride=(args.stride, args.stride),
        needle_mask_threshold=0.6 if not args.full_prostate else -1,
        prostate_mask_threshold=-1 if not args.full_prostate else 0.1,
        transform=SSLTransform(),
    )
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

    ssl_loader = torch.utils.data.DataLoader(
        ssl_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
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

    print(f"SSL Train batches: {len(ssl_loader)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return ssl_loader, train_loader, val_loader, test_loader


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
