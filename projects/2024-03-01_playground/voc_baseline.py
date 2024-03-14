from dataclasses import dataclass


from torchvision.transforms import v2 as T
from torchvision.tv_tensors import Image, Mask
from torchvision.transforms import InterpolationMode
import torch
from torch import nn 
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import one_hot
from einops import rearrange
from monai.metrics.meaniou import MeanIoU, compute_iou
from monai.metrics.cumulative_average import CumulativeAverage
from monai.losses import DiceLoss, DiceCELoss, focal_loss
from skimage.transform import resize
import wandb
from tqdm import tqdm
import os
from enum import StrEnum
from medAI.utils import dataclass_to_dict
from pydantic import BaseModel
from models import HeadBackboneOptimizerConfig, FCN8S, OptimizerConfig


LOG_IMAGE_FREQ = 100


class Args(BaseModel):
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 100
    augment: bool = True

    loss: str = 'ce'

    mask_size: int = 512

    name: str | None = None
    project: str | None = "voc_baseline"
    group: str | None = None
    tags: list | None = None
    id: str | None = None

    opt: OptimizerConfig | HeadBackboneOptimizerConfig = HeadBackboneOptimizerConfig()


def main(args: Args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        name=args.name,
        project=args.project,
        group=args.group,
        tags=args.tags,
        id=args.id,
        config=args.model_dump(),
        dir='/scratch/ssd004/scratch/pwilson'
    )
    wandb.save(__file__)

    # datasets
    from voc_dataset import build_dataset_vector_cluster
    train_ds= build_dataset_vector_cluster(set='train', transform=Transform(augment=args.augment, mask_size=args.mask_size))
    val_ds = build_dataset_vector_cluster(set='val', transform=Transform(augment=False, mask_size=args.mask_size))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    model = FCN8S(21)
    model.to(DEVICE)
    optimizer = model.configure_optimizer(args.opt)

    class_weights = torch.tensor(CLASS_TOTALS).float().reciprocal().to(DEVICE)

    match args.loss:
        case 'focal': 
            from monai.losses.focal_loss import FocalLoss
            criterion = FocalLoss(
                include_background=True,
                to_onehot_y=True,
                use_softmax=True,
            )
        case 'ce':
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        case _: 
            raise ValueError(f'Unknown loss: {args.loss}')

    for epoch in range(args.epochs):
        run_epoch(
            model,
            train_loader,
            criterion,
            DEVICE,
            epoch,
            optimizer,
            "train",
            train=True,
        )
        run_epoch(model, val_loader, criterion, DEVICE, epoch, None, "val", train=False)


""" def build_model(args: ModelArgs):
    match args, args.name: 
        case UNetArgs(), _: 
            from monai.networks.nets import UNet

            return UNet(
                spatial_dims=2, 
                in_channels=3, 
                out_channels=21, 
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2, 2),
            )
        
        case ResNetArgs(), 'simple_resnet_fcn':
            from timm.models.resnet import resnet18

            model = resnet18(pretrained=True)

            model.global_pool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))

            model.fc = torch.nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 21, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            )

            model.layer3[0].conv1.stride = (1, 1)
            model.layer3[0].downsample[0].stride = (1, 1)

            model.layer4[0].conv1.stride = (1, 1)
            model.layer4[0].downsample[0].stride = (1, 1)

            param_groups = [{'lr': args.lr}, {'lr': args.backbone_lr}]
            for name, param in model.named_parameters():
                if 'fc' in name:
                    param_groups[0].append(param)
                else:
                    param_groups[1].append(param)

            return model, param_groups

        case ResNetArgs(), 'simple_resnet_fcn_v2':
            from timm.models.resnet import resnet18

            model = resnet18(pretrained=True)

            model.global_pool = nn.Identity()

            model.fc = torch.nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 21, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            )

            model.layer3[0].conv1.stride = (1, 1)
            model.layer3[0].downsample[0].stride = (1, 1)

            model.layer4[0].conv1.stride = (1, 1)
            model.layer4[0].downsample[0].stride = (1, 1)

            param_groups = [{'lr': args.lr, 'params': []}, {'lr': args.backbone_lr, 'params': []}]
            for name, param in model.named_parameters():
                if 'fc' in name:
                    param_groups[0]['params'].append(param)
                else:
                    param_groups[1]['params'].append(param)

            return model, param_groups
        
        case ResNetArgs(), _:
            from timm.models.resnet import resnet50

            model = resnet50(pretrained=True)
            model.global_pool = torch.nn.Identity()
            model.fc = torch.nn.Identity()
            encoder = model

            decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(2048, 512, 3, 2, 1, 1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(128, 21, 3, 2, 1, 1),
            )

            model = torch.nn.Sequential(encoder, decoder)

        case _, 'vgg_deeplab':

            from timm.models.vgg import vgg16

            vgg = vgg16(pretrained=True)
            
            indices_dilation2 = [17, 19, 21]
            indices_dilation4 = [24, 26, 28]

            def replace_conv_with_dilated_conv(layer, dilation):
                old_conv = vgg.features[layer]
                new_conv = nn.Conv2d(old_conv.in_channels, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, padding=dilation, dilation=dilation)
                new_conv.weight.data = old_conv.weight.data
                new_conv.bias.data = old_conv.bias.data
                vgg.features[layer] = new_conv

            for i in indices_dilation2:
                replace_conv_with_dilated_conv(i, 2)

            for i in indices_dilation4:
                replace_conv_with_dilated_conv(i, 4)

            vgg.features[23] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
            vgg.features[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)

            old_conv = vgg.pre_logits.fc1
            new_conv = nn.Conv2d(512, 4096, 1, 1, 0)
            new_conv.weight.data = old_conv.weight.data.view(4096, 512, 7, 7)[..., 3:4, 3:4]
            new_conv.bias.data = old_conv.bias.data
            vgg.pre_logits.fc1 = new_conv

            vgg.head = nn.Sequential(nn.Conv2d(4096, 21, 1))

            return vgg

        case _, 'vgg_fcn_v0': 
            from timm.models.vgg import vgg16

            model = vgg16(pretrained=True)
            model.head = nn.Sequential(
                nn.Conv2d(4096, 21, 1), 
                nn.Upsample(scale_factor=2, mode='bilinear'), 
                nn.ReLU(), 
                nn.Conv2d(21, 21, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU(),
                nn.Conv2d(21, 21, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear'),
            )
            model.features[23].stride = (1, 1)
            model.features[23] = nn.Sequential(
                model.features[23],
                nn.ZeroPad2d((1, 0, 1, 0))
            )
            model.features[30].stride = (1, 1)
            model.features[30] = nn.Sequential(
                model.features[30],
                nn.ZeroPad2d((1, 0, 1, 0))
            )

            model.features[28].dilation = (2, 2)
            model.features[28].padding = 'same'
            model.features[26].dilation = (2, 2)
            model.features[26].padding = 'same'
            model.features[24].dilation = (2, 2)
            model.features[24].padding = 'same'

            model.pre_logits.fc1.dilation = (4, 4)
            model.pre_logits.fc1.padding = 'same'

            param_groups = [{'lr': args.lr, 'params': []}, {'lr': args.backbone_lr, 'params': []}]
            for name, param in model.named_parameters():
                if 'head' in name:
                    param_groups[0]['params'].append(param)
                else:
                    param_groups[1]['params'].append(param)

            return model, param_groups

        case _: 
            raise ValueError(f'Unknown model: {args.name}')
     """


def run_epoch(
    model, loader, criterion, device, epoch, optimizer=None, prefix="train", train=True
):
    with torch.set_grad_enabled(train):
        model.train(train)

        mean_iou = MeanIoU(include_background=False)
        average_loss = CumulativeAverage()

        for i, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            
            # pred = pred.softmax(1)

            y = one_hot(y, num_classes=256)
            y = rearrange(y, "b h w c -> b c h w")
            y = y[:, :21, :, :]  # only 21 classes are valid components
            y = y.argmax(1)
            loss = criterion(pred, y)

            pred = pred.argmax(1)
            pred = one_hot(pred, num_classes=21)
            pred = rearrange(pred, "b h w c -> b c h w")
            y_ = one_hot(y, num_classes=21).permute(0, 3, 1, 2) > 0

            average_loss.append(loss.item(), x.size(0))
            wandb.log({f"{prefix}/loss": loss.item(), "epoch": epoch})
            mean_iou(pred, y_)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % 100 == 0:
                images = []
                for j in range(len(x)):
                    im = (
                        x[j].permute(1, 2, 0).cpu() * torch.tensor(STD)
                        + torch.tensor(MEAN)
                    ).numpy()

                    mask = torch.nn.functional.interpolate(
                        y[j][None, None, ...].float(), size=im.shape[:2], mode="nearest"
                    )[0,0].cpu().long().numpy()
                    pred_j = torch.nn.functional.interpolate(
                        pred[j][None, ...].float(), size=im.shape[:2], mode="nearest"
                    ).squeeze(0).argmax(0).cpu().numpy()

                    images.append(
                        wandb.Image(
                            im,
                            caption="Image",
                            masks={
                                "ground_truth": {
                                    "mask_data": mask,
                                    "class_labels": CLASS_LABELS,
                                },
                                "prediction": {
                                    "mask_data": pred_j,
                                    "class_labels": CLASS_LABELS,
                                },
                            },
                        ),
                    )
                wandb.log(
                    {
                        f"{prefix}/image": images,
                    }
                )

        wandb.log(
            {
                f"{prefix}/iou": mean_iou.aggregate(),
                "epoch": epoch,
                f"{prefix}/loss_epoch": average_loss.aggregate(),
            }
        )


MEAN = [116.48343652, 112.9986381, 104.11609275]
STD = [60.42993842, 59.48828434, 60.94162154]
CLASS_LABELS = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}
CLASS_TOTALS = [
    74594244,
    467053,
    198893,
    585595,
    407520,
    397646,
    1150645,
    933300,
    1811134,
    766890,
    543177,
    887689,
    1158470,
    620901,
    759544,
    3204420,
    442177,
    590666,
    959799,
    1044949,
    616823,
]


class Transform:
    def __init__(self, augment=False, image_size=512, mask_size=256):
        self.augment = augment
        self.image_size = image_size
        self.mask_size = mask_size

    def __call__(self, image, label):
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255
        label = torch.tensor(np.array(label)).unsqueeze(0).long()
        image = Image(image)
        label = Mask(label)

        H, W = image.shape[-2:]
        image, label = T.CenterCrop(max(H, W))(image, label)
        image, label = T.Resize(
            self.image_size, interpolation=InterpolationMode.BICUBIC
        )(image, label)

        if self.augment:
            image, label = T.RandomHorizontalFlip(p=0.5)(image, label)
            image, label = T.RandomApply(
                [T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)],
                0.3,
            )(image, label)
            image, label = T.RandomResizedCrop(self.image_size, scale=(0.5, 1))(
                image, label
            )

        image = T.Normalize(mean=MEAN, std=STD)(image)
        # label = torch.nn.functional.one_hot(label, num_classes=256)[0].permute(2, 0, 1).float()
        label = T.Resize(self.mask_size, interpolation=InterpolationMode.NEAREST)(
            label
        )[0]
        return image, label


def show(dataset, idx):
    item = dataset[idx]
    npimg = (item[0].permute(1, 2, 0) * torch.tensor(STD) + torch.tensor(MEAN)).numpy()
    plt.imshow(npimg, extent=(0, 1, 1, 0))
    mask = np.array(item[1])
    plt.imshow(
        mask,
        alpha=0.5 * (mask != 0) * (mask != 255),
        extent=(0, 1, 1, 0),
        cmap="tab20",
        vmin=0,
        vmax=21,
    )


if __name__ == "__main__":
    args = Args(
        batch_size=8,
        augment=False,
        epochs=100,
        name=None,
        mask_size=512, 
        opt=OptimizerConfig(lr=1e-4, weight_decay=5e-5, momentum=0.9, optimizer="sgd")
    )

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--nosubmitit", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    _args = parser.parse_args()

    if not _args.nosubmitit:
        from submitit import AutoExecutor, SlurmExecutor

        executor = AutoExecutor(folder="logs")
        executor.update_parameters(
            timeout_min=60 * 8,
            gres="gpu:1",
            cpus_per_task=16,
            mem_gb=16,
            stderr_to_stdout=True,
            slurm_qos='m'
        )
        job = executor.submit(main, args)
        # track the job id
        print("Submitted job:", job.job_id)
        print("Logs and results will be written to:", job.paths.stdout)

    else:
        main(args)
