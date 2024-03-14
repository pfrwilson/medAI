from torchvision.models.vgg import vgg16, VGG16_Weights
import torch 
from torch import nn
from pydantic import BaseModel
import typing as tp


class _VGG16Wrapper(torch.nn.Module):

    FEATS_BY_LAYER = [64, 128, 256, 512, 512]

    def __init__(self):
        super().__init__()
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

    def forward(self, x):
        outs = []
        maxpool_indices = [4, 9, 16, 23, 30]

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in maxpool_indices:
                outs.append(x)

        return outs 


class OptimizerConfig(BaseModel):
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    optimizer: tp.Literal['adam', 'sgd'] = "sgd"


class HeadBackboneOptimizerConfig(OptimizerConfig):
    backbone_lr: tp.Optional[float] = None


class TrainableModel(nn.Module):
    def configure_optimizer(self, cfg: OptimizerConfig):
        if cfg.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


class FCN8S(TrainableModel):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = _VGG16Wrapper()

        self.upsample1 = torch.nn.ConvTranspose2d(512, 512, 3, 2, 1, 1)
        self.upsample2 = torch.nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)
        self.upsample3 = torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)

        self.bn1 = torch.nn.BatchNorm2d(512)
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.clf = torch.nn.Sequential(
            torch.nn.Conv2d(128, num_classes, 1),
            torch.nn.Upsample(scale_factor=4, mode='bilinear')
        )

        self._decoder = torch.nn.ModuleList([
            self.upsample1, self.upsample2, self.upsample3,
            self.bn1, self.bn2, self.bn3, self.clf
        ])

    def forward(self, X):
        feats = self.encoder(X)

        x = self.upsample1(feats[-1])
        x = x + feats[-2]
        x = self.bn1(x).relu()
        x = self.upsample2(x)
        x = x + feats[-3]
        x = self.bn2(x).relu()
        x = self.upsample3(x)
        x = x + feats[-4]
        x = self.bn3(x).relu()

        x = self.clf(x)
        return x

    def configure_optimizer(self, cfg: OptimizerConfig):
        match cfg: 
            case HeadBackboneOptimizerConfig():
                params = [
                    {"params": self.encoder.parameters(), "lr": cfg.backbone_lr},
                    {"params": self._decoder.parameters(), "lr": cfg.lr}
                ]
                if cfg.optimizer == 'adam': 
                    return torch.optim.Adam(params, weight_decay=cfg.weight_decay)
                elif cfg.optimizer == 'sgd':
                    return torch.optim.SGD(params, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
                
            case OptimizerConfig(): 
                if cfg.optimizer == "adam":
                    return torch.optim.Adam(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                elif cfg.optimizer == "sgd":
                    return torch.optim.SGD(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
                
            case _:
                raise ValueError(f"Unknown optimizer config: {cfg}")