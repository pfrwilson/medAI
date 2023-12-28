# from .utils import load_state_dict_from_url
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from dataclasses import dataclass
from .spectral_layers import spectral_norm_conv, spectral_norm_fc, SpectralBatchNorm2d

__all__ = ["SpectralResNet", "resnet18"]  # , 'resnet34', 'resnet50', 'resnet101',
# 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
# 'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):

    """
    This part is from https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/2
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(
        ((h_w + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


def spectral_bn(num_features, coeff=1.0, spectral_bn_active: bool = True):
    if not spectral_bn_active:
        return nn.BatchNorm2d(num_features)
    else:
        return SpectralBatchNorm2d(num_features, coeff)


def spectral_conv(
    in_c,
    out_c,
    kernel_size,
    stride,
    padding: int = 0,
    input_size: int = 256,
    coeff: float = 1.0,
    n_power_iterations: int = 1,
    spectral_conv_active: bool = True,
):
    padding = 1 if kernel_size == 3 else padding

    conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

    if not spectral_conv_active:
        return conv

    if kernel_size == 1:
        # use spectral norm fc, because bound are tight for 1x1 convolutions
        wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
    else:
        # Otherwise use spectral norm conv, with loose bound
        input_dim = (in_c, input_size, input_size)
        wrapped_conv = spectral_norm_conv(conv, coeff, input_dim, n_power_iterations)

    return wrapped_conv


def spectral_fc(
    module,
    coeff: float = 1.0,
    n_power_iterations: int = 1,
    spectral_fc_active: bool = True,
):
    if spectral_fc_active:
        return spectral_norm_fc(module, coeff, n_power_iterations)
    else:
        return module


def spectral_conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    input_size: int = 256,
    coeff: float = 1.0,
    n_power_iterations: int = 1,
    spectral_conv_active: bool = True,
) -> nn.Conv2d:
    """3x3 convolution with padding."""
    conv3x3 = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
    if not spectral_conv_active:
        return conv3x3

    input_dim = (in_planes, input_size, input_size)
    wrapped_conv = spectral_norm_conv(conv3x3, coeff, input_dim, n_power_iterations)

    return wrapped_conv


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


@dataclass
class SpectralConfig:
    input_size: int = 256
    coefficient: float = 1.0
    n_power_iterations: int = 1


class SpectralBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        spectral_config: SpectralConfig = SpectralConfig(),
        spectral_conv_active: bool = True,
    ) -> None:
        super(SpectralBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            if spectral_conv_active:
                norm_layer = lambda num_features: spectral_bn(
                    num_features, spectral_config.coefficient
                )
        if groups != 1 or base_width != 64:
            raise ValueError(
                "SpectralBasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in SpectralBasicBlock"
            )

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = spectral_conv3x3(
            inplanes,
            planes,
            stride,
            input_size=spectral_config.input_size,
            coeff=spectral_config.coefficient,
            n_power_iterations=spectral_config.n_power_iterations,
            spectral_conv_active=spectral_conv_active,
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        # output size of previous layer used as input for new layer
        self.output_size_, _ = conv_output_shape(
            spectral_config.input_size,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        self.conv2 = spectral_conv3x3(
            planes,
            planes,
            input_size=self.output_size_,
            coeff=spectral_config.coefficient,
            n_power_iterations=spectral_config.n_power_iterations,
            spectral_conv_active=spectral_conv_active,
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # final output size of block
        self.output_size_, _ = conv_output_shape(
            self.output_size_,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    @property
    def output_size(self):
        return self.output_size_

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as SpectralResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # TODO: this would not work
        self.conv2 = spectral_conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SpectralResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[SpectralBasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        in_channels: int = 3,
        spectral_config: SpectralConfig = SpectralConfig(),
        spectral_norm: bool = True,
    ) -> None:

        super(SpectralResNet, self).__init__()

        self.spectral_config = spectral_config
        #print(spectral_config)
        self.spectral_norm = spectral_norm

        norm_layer = lambda num_features: spectral_bn(
            num_features, spectral_config.coefficient, spectral_bn_active=False
        )
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = spectral_conv(
            in_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            input_size=spectral_config.input_size,
            coeff=spectral_config.coefficient,
            n_power_iterations=spectral_config.n_power_iterations,
            spectral_conv_active=spectral_norm,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # finding output size to use as input size for the next layer based on forward pass
        self.output_size, _ = conv_output_shape(
            spectral_config.input_size, kernel_size=7, stride=2, padding=3
        )
        self.output_size, _ = conv_output_shape(
            self.output_size, kernel_size=3, stride=2, padding=1
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = spectral_fc(
            nn.Linear(512 * block.expansion, num_classes),
            coeff=spectral_config.coefficient,
            n_power_iterations=spectral_config.n_power_iterations,
            spectral_fc_active=spectral_norm,
        )
        # self.fc = IsoMaxLossFirstPart(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, SpectralBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[SpectralBasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                spectral_fc(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    coeff=self.spectral_config.coefficient,
                    n_power_iterations=self.spectral_config.n_power_iterations,
                    spectral_fc_active=self.spectral_norm,
                ),
                norm_layer(planes * block.expansion),
            )

        # spectral input size for the next layer
        self.spectral_config.input_size = self.output_size

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                spectral_config=self.spectral_config,
                spectral_conv_active=self.spectral_norm,
            )
        )

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # output size for the next layer
            self.output_size = layers[-1].output_size
            # spectral input size for the next layer
            self.spectral_config.input_size = self.output_size

            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    spectral_config=self.spectral_config,
                    spectral_conv_active=self.spectral_norm,
                )
            )

        # output size for the next block
        self.output_size = layers[-1].output_size
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # return x[:, :2]
        return x

    def forward(self, x: Tensor, *args) -> Tensor:
        return self._forward_impl(x)


class ResNetCustomChannels(nn.Module):
    def __init__(
        self,
        block: Type[Union[SpectralBasicBlock, Bottleneck]],
        layers: List[int],
        layer_channels: List[int] = [64, 128, 256, 512],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        in_channels=3,
    ) -> None:
        super(ResNetCustomChannels, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        assert (
            len(layer_channels) == 4 and len(layers) == 4
        ), "Must have exactly 4 sections"

        self.inplanes = layer_channels[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layer_channels[0], layers[0])
        self.layer2 = self._make_layer(
            block,
            layer_channels[1],
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            layer_channels[2],
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            layer_channels[3],
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes * block.expansion, num_classes)
        # self.fc = IsoMaxLossFirstPart(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, SpectralBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[SpectralBasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # return x[:, :2]
        return x

    def forward(self, x: Tensor, *args) -> Tensor:
        return self._forward_impl(x)


## code from https://github.com/aredier/monte_carlo_dropout/blob/5271f0b45c855958a243e997c1d6860dc6ee874e/monte_carlo_dropout/SpectralResNet.py
class DropoutBlock(nn.Module):
    """same as a basic block but adding dropout to it."""

    def __init__(self, basic_block: SpectralBasicBlock, dropout_rate: float = 0.0):
        super(DropoutBlock, self).__init__()
        self.conv1 = basic_block.conv1
        self.bn1 = basic_block.bn1
        self.relu = basic_block.relu
        self.conv2 = basic_block.conv2
        self.bn2 = basic_block.bn2
        self.downsample = basic_block.downsample
        self.stride = basic_block.stride
        ## todo: check for always dropout
        self.force_dropout = True
        self.dropout_rate = dropout_rate

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(
            out, p=self.dropout_rate, training=self.training or self.force_dropout
        )
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.dropout(
            out, p=self.dropout_rate, training=self.training or self.force_dropout
        )

        out += identity
        out = self.relu(out)

        return out


class DropoutResnet(nn.Module):
    """adds dropout to an existing SpectralResNet."""

    def __init__(self, source_resnet: SpectralResNet, dropout_rate: float = 0.0):

        super(DropoutResnet, self).__init__()
        self._norm_layer = source_resnet._norm_layer

        self.inplanes = source_resnet.inplanes
        self.dilation = source_resnet.dilation
        self.groups = source_resnet.groups
        self.base_width = source_resnet.base_width
        self.conv1 = source_resnet.conv1
        self.bn1 = source_resnet.bn1
        self.relu = source_resnet.relu
        self.maxpool = source_resnet.maxpool
        self.layer1 = self._make_layer(source_resnet.layer1, dropout_rate)
        self.layer2 = self._make_layer(source_resnet.layer2, dropout_rate)
        self.layer3 = self._make_layer(source_resnet.layer3, dropout_rate)
        self.layer4 = self._make_layer(source_resnet.layer4, dropout_rate)
        self.avgpool = source_resnet.avgpool
        self.fc = source_resnet.fc

    @staticmethod
    def _set_force_dropout_on_layer(force_dropout: bool, layer: nn.Sequential):
        for block in layer.children():
            block.force_dropout = force_dropout

    def set_force_dropout(self, force_dropout):
        self._set_force_dropout_on_layer(force_dropout, self.layer1)
        self._set_force_dropout_on_layer(force_dropout, self.layer2)
        self._set_force_dropout_on_layer(force_dropout, self.layer3)
        self._set_force_dropout_on_layer(force_dropout, self.layer4)

    def _make_layer(self, source_layer: nn.Sequential, dropout_rate):
        return nn.Sequential(
            *[DropoutBlock(block, dropout_rate) for block in source_layer.children()]
        )

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x, *args):
        return self._forward(x)


def _spectral_resnet(
    arch: str,
    block: Type[Union[SpectralBasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    drop_rate="none",
    **kwargs: Any
):
    model = SpectralResNet(block, layers, spectral_config=SpectralConfig(), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    if drop_rate != "none":
        return DropoutResnet(source_resnet=model, dropout_rate=drop_rate)
    return model


def spectral_resnet50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> SpectralResNet:
    r"""SpectralResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _spectral_resnet(
        "spectral_resnet50",
        SpectralBasicBlock,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs
    )


def spectral_resnet18(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> SpectralResNet:
    r"""SpectralResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _spectral_resnet(
        "spectral_resnet18",
        SpectralBasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        **kwargs
    )


def spectral_resnet10(
    pretrained: bool = False, progress: bool = False, drop_rate="none", **kwargs: Any
):
    r"""SpectralResNet-10 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if drop_rate != "none":
        return _spectral_resnet(
            "resnet10",
            SpectralBasicBlock,
            [1, 1, 1, 1],
            pretrained,
            progress,
            drop_rate=drop_rate,
            **kwargs
        )
    return _spectral_resnet(
        "spectral_resnet10",
        SpectralBasicBlock,
        [1, 1, 1, 1],
        pretrained,
        progress,
        drop_rate=drop_rate,
        **kwargs
    )


def resnet10_custom(
    in_channels,
    n_classes,
    layer_channels=[64, 128, 256, 512],
    drop_rate="none",
):
    model = ResNetCustomChannels(
        SpectralBasicBlock,
        [1, 1, 1, 1],
        layer_channels=layer_channels,
        in_channels=in_channels,
        num_classes=n_classes,
    )

    if drop_rate != "none":
        return DropoutResnet(model, drop_rate)
    else:
        return model


def get_features(resnet_model, X):
    return resnet_model(X)


def spectral_resnet_feature_extractor(spectral_resnet_model: SpectralResNet):
    """
    Makes the given SpectralResNet model support the feature extraction protocol
    by monkey patching it and removing the fully connected layer
    """
    feature_dim = spectral_resnet_model.fc.in_features
    spectral_resnet_model.fc = torch.nn.Identity()
    spectral_resnet_model.features_dim = feature_dim

    spectral_resnet_model.__class__.get_features = get_features
    return spectral_resnet_model