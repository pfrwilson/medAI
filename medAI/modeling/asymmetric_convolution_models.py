from torch import nn 
import torch 
from dataclasses import dataclass, field
from simple_parsing import choice
import typing as tp 


class VerticalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(VerticalConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(1, 0),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(1, 1),
            padding=(1, 0),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample_identity = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(stride, 1),
                bias=False,
            )
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample_identity is not None:
            identity = self.downsample_identity(identity)
        out += identity
        out = self.relu(out)
        return out


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample_identity = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(stride, stride),
                bias=False,
            )
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample_identity is not None:
            identity = self.downsample_identity(identity)
        out += identity
        out = self.relu(out)
        return out


class RFLineResNet(nn.Module):
    def __init__(
        self,
        layers=[2, 2, 2, 2],
        channels=[16, 32, 64, 128, 256],
        convs=["vertical", "vertical", "vertical", "basic"],
        num_classes=2,
        in_channels=1,
    ):
        super(RFLineResNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            channels[0],
            kernel_size=(7, 3),
            padding=(3, 1),
            stride=2,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(channels[0])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

        self.layers = []
        for layer in range(len(layers)):
            self._make_layer(
                channels[layer],
                channels[layer + 1],
                layers[layer],
                stride=1 if layer == 0 else 2,
                block=VerticalConvBlock
                if convs[layer] == "vertical"
                else BasicConvBlock,
            )

        self.layers = nn.Sequential(*self.layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def _make_layer(
        self, in_channels, out_channels, n_blocks, stride=1, block=BasicConvBlock
    ):
        blocks = []
        blocks.append(block(in_channels, out_channels, kernel_size=3, stride=stride))
        for _ in range(1, n_blocks):
            blocks.append(block(out_channels, out_channels, kernel_size=3))

        self.layers.append(nn.Sequential(*blocks))

    def forward(self, x):
        # H, W is 447, 56

        x = self.conv1(x)  # 447 -> 224, 56 -> 28
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 224 -> 112, 28 -> 28

        x = self.layers(x)

        x = self.avgpool(x)  # 112 -> 1, 28 -> 1
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        # now H, W is


SUPPORTED_MODELS = [
        "resnet10",
        "resnet10_small",
        "rfline_resnet",
        "rfline_resnet_small",
    ]


@dataclass
class ModelConfig:
    """Configuration for the model factory."""
    model_name: str = choice(SUPPORTED_MODELS, default="resnet10")
    # model_args: tp.Dict[str, tp.Any] = field(default_factory=dict)


def create_model(args: ModelConfig):
    if args.model_name == "resnet10":
        from trusnet.modeling.registry import resnet10

        model = resnet10()

    elif args.model_name == "rfline_resnet":
        model = RFLineResNet()

    elif args.model_name == "rfline_resnet_small":
        model = RFLineResNet(layers=[1, 1, 1, 1])

    elif args.model_name == "resnet10_small":
        from trusnet.modeling.resnets import ResNetCustomChannels, BasicBlock

        model = ResNetCustomChannels(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
            layer_channels=[32, 64, 128, 256],
            num_classes=2,
            norm_layer=nn.BatchNorm2d,
            in_channels=1,
        )

    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    return model

