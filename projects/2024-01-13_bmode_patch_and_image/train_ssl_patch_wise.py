"""Runs the training of the SSL model on the patch-wise data.



"""


import argparse
import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from train_supervised_patch_wise import (
    patchwise_train_loop,
    patchwise_evaluation_loop,
    evaluation_loop,
    SlidingWindowModel,
)
from medAI.utils.common import EarlyStopping
import logging
import rich_argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def parse_args():
    class HelpFormatter(
        rich_argparse.ArgumentDefaultsRichHelpFormatter,
        rich_argparse.MetavarTypeRichHelpFormatter,
    ):
        ...

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter, add_help=False, description=__doc__
    )

    group = parser.add_argument_group("General")
    group.add_argument("--name", type=str, default="ssl_patch_wise")
    group.add_argument("--group", type=str, default=None)
    group.add_argument("--debug", action="store_true", default=False)

    # Data
    group = parser.add_argument_group("Data")
    group.add_argument("--batch-size", type=int, default=32, help="batch size for SSL")
    group.add_argument("--patch-size", type=int, default=128, help="patch size")
    group.add_argument("--stride", type=int, default=32, help="stride")
    group.add_argument(
        "--pre-crop-patch-size-ratio",
        type=float,
        default=1.5,
        help="Ratio of the patch size to the pre-crop size",
    )
    group.add_argument(
        "--needle-mask-threshold",
        type=float,
        default=-1,
        help="Needle mask threshold to select training patches",
    )
    group.add_argument(
        "--prostate-mask-threshold",
        type=float,
        default=0.9,
        help="Prostate mask threshold to select training patches",
    )
    group.add_argument(
        "--benign-to-cancer-ratio-train",
        type=float,
        default=1,
        help="Ratio of benign to cancer patches in the training set (SSL)",
    )

    group = parser.add_argument_group("Training")
    group.add_argument("--epochs", type=int, default=30)
    group.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="learning rate. We previously found 1e-4 to work well.",
    )
    group.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="L2 weight decay",
    )

    parser.add_argument("-h", action="help", help="show this help message and exit")
    return parser.parse_args()


def main(args):
    logging.info(f"Running experiment {args.name}")

    wandb.init(
        project="2024-01-13_bmode_patch_and_image",
        name=args.name,
        group=args.group,
        config=args,
    )

    logging.info("Loading data")
    (
        patch_loader_ssl,
        train_patch_loader,
        val_patch_loader,
        test_patch_loader,
        train_image_loader,
        val_image_loader,
        test_image_loader,
    ) = dataloaders(
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        pre_crop_patch_size_ratio=args.pre_crop_patch_size_ratio,
        needle_mask_threshold=args.needle_mask_threshold,
        prostate_mask_threshold=args.prostate_mask_threshold,
        benign_to_cancer_ratio_train=args.benign_to_cancer_ratio_train,
    )
    logging.info("Data loaded")
    logging.info(f"Number of self-supervised patches: {len(patch_loader_ssl.dataset)}")
    logging.info(f"Number of supervised patches: {len(train_patch_loader.dataset)}")
    logging.info(f"Number of training images: {len(train_image_loader.dataset)}")
    logging.info(f"Number of validation images: {len(val_image_loader.dataset)}")
    logging.info(f"Number of test images: {len(test_image_loader.dataset)}")

    # model
    logging.info("Creating model")
    from timm.models.resnet import resnet10t

    backbone = resnet10t(in_chans=1, num_classes=1)
    backbone.fc = torch.nn.Identity()

    backbone = torch.nn.Sequential(torch.nn.InstanceNorm2d(1), backbone)

    from medAI.modeling.vicreg import VICReg

    ssl_model = VICReg(backbone, proj_dims=[512, 512, 512], features_dim=512)

    logging.info("Training")
    optimizer = torch.optim.Adam(
        ssl_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    from medAI.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=5,
        max_epochs=args.epochs,
        warmup_start_lr=args.lr / 10,
    )

    best_score = 0
    best_model_state = None

    for epoch in range(10):
        print(f"Epoch {epoch}")
        ssl_train_epoch(ssl_model, optimizer, patch_loader_ssl)
        scheduler.step()

        best_score, best_model_state = fit_linear_probe(
            backbone,
            features_dim=512,
            train_loader=train_patch_loader,
            val_loader=val_patch_loader,
            test_loader=test_patch_loader,
            epochs=20,
            patch_size=args.patch_size,
            stride=args.stride,
            best_score=best_score,
            best_model_state=best_model_state,
        )
        wandb.log({"best_val_auc_high_involvement": best_score})


def ssl_train_epoch(model, optimizer, loader):
    """
    Perform a single epoch of semi-supervised learning training.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loader (torch.utils.data.DataLoader): The data loader for loading the training data.

    Returns:
        None
    """
    model.train()
    model.cuda()

    for batch_idx, (patch1, patch2) in enumerate(tqdm(loader, desc="SSL training")):
        patch1, patch2 = patch1.cuda(), patch2.cuda()
        optimizer.zero_grad()
        loss = model(patch1, patch2)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            wandb.log({"ssl_loss": loss.item()})


def fit_linear_probe(
    backbone,
    features_dim,
    train_loader,
    val_loader,
    test_loader,
    epochs=10,
    patch_size=128,
    stride=64,
    best_score=0,
    best_model_state=None,
):
    """
    Fits a linear probe model on top of a backbone network using patch-wise training.

    Args:
        backbone (nn.Module): The backbone network.
        features_dim (int): The dimension of the features extracted by the backbone network.
        train_loader (DataLoader): The data loader for the training set.
        val_loader (DataLoader): The data loader for the validation set.
        epochs (int, optional): The number of training epochs. Defaults to 10.
        patch_size (int, optional): The size of the patches used for training. Defaults to 128.
        stride (int, optional): The stride used for patch extraction. Defaults to 64.
        best_score (float, optional): The best score achieved during training. Defaults to 0.
        best_model_state (dict, optional): The state dictionary of the best model achieved during training. Defaults to None.

    Returns:
        tuple: A tuple containing the best score achieved during training and the state dictionary of the best model.
    """
    linear_probe = LinearProbe(backbone, features_dim, 2).cuda()
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=0.0001)
    hmap_model = SlidingWindowModel(linear_probe, patch_size, stride).cuda()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    monitor = EarlyStopping(mode='max', patience=3)

    for epoch in range(epochs):
        patchwise_train_loop(
            train_loader, linear_probe, torch.nn.CrossEntropyLoss(), optimizer
        )
        scheduler.step()
        metrics = patchwise_evaluation_loop(val_loader, linear_probe, log_prefix="val_")
        score = metrics["auc_high_involvement"]

        monitor(score)
        if monitor.early_stop:
            break
        if score > best_score:
            best_score = score
            best_model_state = linear_probe.state_dict()
            patchwise_evaluation_loop(test_loader, linear_probe, log_prefix="test_")

    return best_score, best_model_state


class LinearProbe(torch.nn.Module):
    """Linear probe for the SSL model.

    Freeze the backbone and train a linear model on top of the features.
    """

    def __init__(self, backbone, features_dim, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.linear = torch.nn.Linear(features_dim, num_classes)
        self.freeze_backbone = freeze_backbone

    def forward(self, x):
        if self.freeze_backbone: 
            with torch.no_grad():
                self.backbone.eval()
                x = self.backbone(x)
        else: 
            x = self.backbone(x)
        return self.linear(x)


def dataloaders(
    batch_size=32,
    patch_size=128,
    stride=64,
    pre_crop_patch_size_ratio=1.5,
    needle_mask_threshold=-1,
    prostate_mask_threshold=0.9,
    benign_to_cancer_ratio_train=1,
):
    from medAI.datasets.nct2013.nctbmode1024px import (
        BModePatches,
        BModeImages,
    )
    from train_supervised_patch_wise import image_dataloaders, patch_dataloaders
    from medAI.datasets.nct2013 import KFoldPatientSelector, CoresSelector

    patient_selector = KFoldPatientSelector()
    train_image_loader, val_image_loader, test_image_loader = image_dataloaders(
        patient_selector=patient_selector,
    )
    train_patch_loader, val_patch_loader, test_patch_loader = patch_dataloaders(
        patient_selector=patient_selector,
        batch_size=batch_size,
        patch_size=patch_size,
        min_involvement_train=40, 
        prostate_mask_threshold=0.9, 
        needle_mask_threshold=0.6
    )

    def ssl_transform(item):
        patch = item["patch"]
        patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        patch = patch / 255.0

        from torchvision import transforms as T

        augmentations = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(patch_size),
                T.RandomVerticalFlip(p=0.5),
            ]
        )
        patch = augmentations(patch)
        patch2 = augmentations(patch)
        return patch, patch2

    dataset_ssl = BModePatches(
        split="train",
        cores_selector=CoresSelector(
            patient_selector=patient_selector,
            benign_to_cancer_ratio=benign_to_cancer_ratio_train,
        ),
        transform=ssl_transform,
        patch_size=int(patch_size * pre_crop_patch_size_ratio),
        stride=stride,
        needle_mask_threshold=needle_mask_threshold,
        prostate_mask_threshold=prostate_mask_threshold,
    )
    fig, ax = plt.subplots(3, 3)
    for i in range(9):
        dataset_ssl.show_patch_extraction(ax[i // 3, i % 3])

    if wandb.run is not None:
        wandb.log({"SSL patch_extraction": wandb.Image(fig)})

    patch_loader_ssl = torch.utils.data.DataLoader(
        dataset_ssl,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    return (
        patch_loader_ssl,
        train_patch_loader,
        val_patch_loader,
        test_patch_loader,
        train_image_loader,
        val_image_loader,
        test_image_loader,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
