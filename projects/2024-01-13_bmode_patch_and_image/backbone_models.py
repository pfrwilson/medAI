import torch


__all__ = [
    "resnet10t",
    "resnet10t_norm_input",
    "resnet10t_gn",
    "resnet10t_gn_norm_input",
    "resnet10t_layer_norm",
    "resnet10t_isomax",
    "resnet10t_norm_input_isomax",
]


def resnet10t():
    from timm.models.resnet import resnet10t

    model = resnet10t(in_chans=1, num_classes=1)
    return model


def resnet10t_norm_input():
    from timm.models.resnet import resnet10t

    model = resnet10t(in_chans=1, num_classes=1)
    model = torch.nn.Sequential(torch.nn.InstanceNorm2d(1), model)

    return model


def resnet10t_gn_norm_input():
    from timm.models.resnet import resnet10t

    model = resnet10t(
        in_chans=1,
        num_classes=1,
        norm_layer=lambda chans: torch.nn.GroupNorm(8, chans),
    )
    model = torch.nn.Sequential(torch.nn.InstanceNorm2d(1), model)

    return model


def resnet10t_gn():
    from timm.models.resnet import resnet10t

    model = resnet10t(
        in_chans=1,
        num_classes=1,
        norm_layer=lambda chans: torch.nn.GroupNorm(8, chans),
    )
    return model


def resnet10t_layer_norm():
    from timm.models.resnet import resnet10t
    from medAI.modeling.common import LayerNorm2d

    model = resnet10t(
        in_chans=1, num_classes=1, norm_layer=lambda chans: LayerNorm2d(chans)
    )

    return model


def resnet10t_isomax():
    backbone = resnet10t()
    from medAI.modeling.isomax_plus import (
        IsoMaxPlusLossFirstPart,
        IsoMaxPlusLossSecondPart,
    )
    backbone.fc = IsoMaxPlusLossFirstPart(num_features=512, num_classes=2)

    return backbone


def resnet10t_norm_input_isomax():
    backbone = resnet10t_norm_input()
    from medAI.modeling.isomax_plus import (
        IsoMaxPlusLossFirstPart,
        IsoMaxPlusLossSecondPart,
    )
    backbone[0].fc = IsoMaxPlusLossFirstPart(num_features=512, num_classes=2)

    return backbone