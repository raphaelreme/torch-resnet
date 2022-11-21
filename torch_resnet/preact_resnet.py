"""Pre-activation resnets

Note that in [3], Wide ResNet 50-2 is without pre-activation contrary to Wide ResNet16/28/40
"""

from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .avg_2d import Avg2d


__all__ = [
    "PreActResNet",
    "PreActResNet18",
    "PreActResNet34",
    "PreActResNet50",
    "PreActResNet101",
    "PreActResNet152",
    "PreActResNet200",
    "PreActWideResNet50_2",
    "PreActWideResNet101_2",
    "PreActResNet20",
    "PreActResNet32",
    "PreActResNet44",
    "PreActResNet56",
    "PreActResNet110",
    "PreActResNet164",
    "PreActResNet1001",
    "PreActResNet1202",
    "PreActWideResNet16",
    "PreActWideResNet28",
    "PreActWideResNet40",
]


class PreActBlock(nn.Module):
    """Pre-activation basic bloc"""

    def __init__(self, in_planes: int, planes: int, stride=1, width=1, drop_rate=0.0):
        """Constructor

        Args:
            in_planes (int): Input dimension of features
            planes (int): Default target dimension of features (before applying width or expansion)
                See self.out_planes to know the real output dimension
            stride (int): Stride of the first convolution to downsample spatial dimensions
                usally set to 1 (no downsampling) or 2 (downsample by 2)
            width (int): Width of the block for WideResNet.
            drop_rate (float): dropout rate to apply between the convolution.
        """
        super().__init__()
        self.out_planes = planes * width

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, self.out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(drop_rate, inplace=True)
        self.bn2 = nn.BatchNorm2d(self.out_planes)
        self.conv2 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.out_planes:
            self.shortcut = nn.Conv2d(in_planes, self.out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        # When shortcut is not identity, let's apply shortcut on the activated input
        # Otherwise let's have a real skip connection
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(self.dropout(F.relu(self.bn2(out))))  # Dropout after relu, and before second conv cf [3]
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation bottleneck

    Expand features dimension by expansion
    """

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride=1, width=1, drop_rate=0.0):
        """Constructor

        Args:
            in_planes (int): Input dimension of features
            planes (int): Default target dimension of features (before applying width or expansion)
                See self.out_planes to know the real output dimension
            stride (int): Stride of the 3x3 convolution to downsample spatial dimensions
                usally set to 1 (no downsampling) or 2 (downsample by 2)
            width (int): Width of the block for WideResNet. (Increase only 3x3 conv features)
            drop_rate (float): Unused. Kept to match the BasicBlock api.
        """
        super().__init__()
        self.out_planes = planes * self.expansion
        mid_planes = planes * width

        if drop_rate != 0:
            raise ValueError("Dropout in bottleneck is not supported")

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, self.out_planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.out_planes:
            self.shortcut = nn.Conv2d(in_planes, self.out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        # When shortcut is not identity, let's apply shortcut on the activated input
        # Otherwise let's have a real skip connection
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    """Base class for Resnet with pre-activation"""

    def __init__(
        self,
        block: type,
        dimensions: List[Tuple[int, int]],
        in_planes=3,
        width=1,
        drop_rate=0.0,
        small_images=False,
    ):
        """Constructor

        Args:
            block (type): Constructor of a block (PreActBlock or Bottleneck or your own)
            dimensions (List[Tuple[int, int]]): All the dimensions of the resnet as (num_blocks, planes)
                for each main layer.
            in_planes (int): Number of channels in the input images
                Default: 3 (standard images)
            width (int): Width of Wide-Resnet
                Default: 1 (Standard Resnet)
            drop_rate (float): Dropout rate for Wide-Resnet with basic block
            small_images (bool): With small images (size < 100px), let's rather use Cifar version of ResNet
        """
        super().__init__()

        self.out_planes = dimensions[0][1]

        self.init_layer: nn.Module
        if small_images:
            self.init_layer = nn.Conv2d(in_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.init_layer = nn.Sequential(
                nn.Conv2d(in_planes, self.out_planes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        strides = [1] + [2] * (len(dimensions) - 2)
        layers: List[nn.Module] = []
        for (num_blocks, planes), stride in zip(dimensions[1:], strides):
            layers.append(self._make_layer(block, planes, num_blocks, stride, width, drop_rate))

        self.layers = nn.Sequential(*layers)
        self.norm = nn.BatchNorm2d(self.out_planes)
        self.avgpool = Avg2d()
        self.head: nn.Module = nn.Identity()

    def _make_layer(
        self, block: type, planes: int, num_blocks: int, stride: int, width: int, drop_rate: float
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_ in strides:
            layers.append(block(self.out_planes, planes, stride_, width, drop_rate))
            self.out_planes = layers[-1].out_planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.init_layer(x)
        out = self.layers(out)
        out = F.relu(self.norm(out))
        out = self.avgpool(out)
        return self.head(out)

    def set_head(self, head: nn.Module) -> None:
        """Set the head of the ResNet

        By default the head is an identity after the AvgPooling

        For classification a simple linear head is enough:
        ```python
            model.set_head(nn.Linear(model.out_planes, num_classes))
        ```

        Args:
            head (nn.Module): Head module
        """
        self.head = head


# Designed for ImageNet-1000


class PreActResNet18(PreActResNet):
    """Resnet 18 with preactivation

    Was designed for ImageNet but can be adapted to cifar with small_images parameter
    """

    def __init__(self, in_planes=3, small_images=False):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
            small_images (bool): With small images (size < 100px), let's rather use Cifar version of ResNet
        """
        super().__init__(
            PreActBlock, list(zip([1, 2, 2, 2, 2], [64, 64, 128, 256, 512])), in_planes, small_images=small_images
        )


class PreActResNet34(PreActResNet):
    """Resnet 34 with preactivation

    Was designed for ImageNet but can be adapted to cifar with small_images parameter
    """

    def __init__(self, in_planes=3, small_images=False):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
            small_images (bool): With small images (size < 100px), let's rather use Cifar version of ResNet
        """
        super().__init__(
            PreActBlock, list(zip([1, 3, 4, 6, 3], [64, 64, 128, 256, 512])), in_planes, small_images=small_images
        )


class PreActResNet50(PreActResNet):
    """Resnet 50 with preactivation

    Was designed for ImageNet but can be adapted to Cifar with small_images parameter
    """

    def __init__(self, in_planes=3, small_images=False):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
            small_images (bool): With small images (size < 100px), let's rather use Cifar version of ResNet
        """
        super().__init__(
            PreActBottleneck, list(zip([1, 3, 4, 6, 3], [64, 64, 128, 256, 512])), in_planes, small_images=small_images
        )


class PreActResNet101(PreActResNet):
    """Resnet 101 with preactivation

    Was designed for ImageNet but can be adapted to Cifar with small_images parameter
    """

    def __init__(self, in_planes=3, small_images=False):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
            small_images (bool): With small images (size < 100px), let's rather use Cifar version of ResNet
        """
        super().__init__(
            PreActBottleneck, list(zip([1, 3, 4, 23, 3], [64, 64, 128, 256, 512])), in_planes, small_images=small_images
        )


class PreActResNet152(PreActResNet):
    """Resnet 152 with preactivation

    Was designed for ImageNet but can be adapted to Cifar with small_images parameter
    """

    def __init__(self, in_planes=3, small_images=False):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
            small_images (bool): With small images (size < 100px), let's rather use Cifar version of ResNet
        """
        super().__init__(
            PreActBottleneck, list(zip([1, 3, 8, 36, 3], [64, 64, 128, 256, 512])), in_planes, small_images=small_images
        )


class PreActResNet200(PreActResNet):
    """Resnet 200 with preactivation

    Was designed for ImageNet but can be adapted to Cifar with small_images parameter
    """

    def __init__(self, in_planes=3, small_images=False):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
            small_images (bool): With small images (size < 100px), let's rather use Cifar version of ResNet
        """
        super().__init__(
            PreActBottleneck,
            list(zip([1, 3, 24, 36, 3], [64, 64, 128, 256, 512])),
            in_planes,
            small_images=small_images,
        )


class PreActWideResNet50_2(PreActResNet):  # pylint: disable=invalid-name
    """WideResnet 50-2 with preactivation

    Was designed for ImageNet but can be adapted to Cifar with small_images parameter
    """

    def __init__(self, in_planes=3, small_images=False):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
            small_images (bool): With small images (size < 100px), let's rather use Cifar version of ResNet
        """
        super().__init__(
            PreActBottleneck,
            list(zip([1, 3, 4, 6, 3], [64, 64, 128, 256, 512])),
            in_planes,
            width=2,
            small_images=small_images,
        )


class PreActWideResNet101_2(PreActResNet):  # pylint: disable=invalid-name
    """WideResnet 101-2 with preactivation

    Was designed for ImageNet but can be adapted to Cifar with small_images parameter
    """

    def __init__(self, in_planes=3, small_images=False):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
            small_images (bool): With small images (size < 100px), let's rather use Cifar version of ResNet
        """
        super().__init__(
            PreActBottleneck, list(zip([1, 3, 4, 23, 3], [64, 64, 128, 256, 512])), in_planes, small_images=small_images
        )


# Designed for Cifar


class PreActResNet20(PreActResNet):
    """Resnet 20 with preactivation

    Was designed for Cifar.
    """

    def __init__(self, in_planes=3):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
        """
        super().__init__(PreActBlock, list(zip([1, 3, 3, 3], [16, 16, 32, 64])), in_planes, small_images=True)


class PreActResNet32(PreActResNet):
    """Resnet 32 with preactivation

    Was designed for Cifar.
    """

    def __init__(self, in_planes=3):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
        """
        super().__init__(PreActBlock, list(zip([1, 5, 5, 5], [16, 16, 32, 64])), in_planes, small_images=True)


class PreActResNet44(PreActResNet):
    """Resnet 44 with preactivation

    Was designed for Cifar.
    """

    def __init__(self, in_planes=3):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
        """
        super().__init__(PreActBlock, list(zip([1, 7, 7, 7], [16, 16, 32, 64])), in_planes, small_images=True)


class PreActResNet56(PreActResNet):
    """Resnet 56 with preactivation

    Was designed for Cifar.
    """

    def __init__(self, in_planes=3):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
        """
        super().__init__(PreActBlock, list(zip([1, 9, 9, 9], [16, 16, 32, 64])), in_planes, small_images=True)


class PreActResNet110(PreActResNet):
    """Resnet 110 with preactivation

    Was designed for Cifar.
    """

    def __init__(self, in_planes=3):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
        """
        super().__init__(PreActBlock, list(zip([1, 18, 18, 18], [16, 16, 32, 64])), in_planes, small_images=True)


class PreActResNet164(PreActResNet):
    """Resnet 164 with preactivation

    Was designed for Cifar.
    """

    def __init__(self, in_planes=3):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
        """
        # Following [2] -> Usage of bottle neck
        super().__init__(PreActBottleneck, list(zip([1, 18, 18, 18], [16, 16, 32, 64])), in_planes, small_images=True)


class PreActResNet1001(PreActResNet):
    """Resnet 1001 with preactivation

    Was designed for Cifar.
    """

    def __init__(self, in_planes=3):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
        """
        # Following [2] -> Usage of bottle neck
        super().__init__(
            PreActBottleneck, list(zip([1, 111, 111, 111], [16, 16, 32, 64])), in_planes, small_images=True
        )


class PreActResNet1202(PreActResNet):
    """Resnet 1202 with preactivation

    Was designed for Cifar. Performs poorly according to [1].
    """

    def __init__(self, in_planes=3):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
        """
        # Following [1] -> Default block
        super().__init__(PreActBlock, list(zip([1, 200, 200, 200], [16, 16, 32, 64])), in_planes, small_images=True)


class PreActWideResNet16(PreActResNet):
    """WideResnet 16-k with preactivation

    Was designed for Cifar.

    The number of layer matches the paper [3] and does not follow [1] and [2]:
    The shortcut layers (2 for these models) are added to the count.
    """

    def __init__(self, in_planes=3, width=1, drop_rate=0.3):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
            width (int): width of the resnet
                Default: 1 (No width <=> PreActResNet14)
            drop_date (float): Dropout rate in wide basic block
        """
        super().__init__(
            PreActBlock,
            list(zip([1, 2, 2, 2], [16, 16, 32, 64])),
            in_planes,
            width=width,
            drop_rate=drop_rate,
            small_images=True,
        )


class PreActWideResNet28(PreActResNet):
    """WideResnet 28-k with preactivation

    Was designed for Cifar.

    The number of layer matches the paper [3] and does not follow [1] and [2]:
    The shortcut layers (2 for these models) are added to the count.
    """

    def __init__(self, in_planes=3, width=1, drop_rate=0.3):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
            width (int): width of the resnet
                Default: 1 (No width <=> PreActResNet26)
            drop_date (float): Dropout rate in wide basic block
        """
        super().__init__(
            PreActBlock,
            list(zip([1, 4, 4, 4], [16, 16, 32, 64])),
            in_planes,
            width=width,
            drop_rate=drop_rate,
            small_images=True,
        )


class PreActWideResNet40(PreActResNet):
    """WideResnet 40-k with preactivation

    Was designed for Cifar.

    The number of layer matches the paper [3] and does not follow [1] and [2]:
    The shortcut layers (2 for these models) are added to the count.
    """

    def __init__(self, in_planes=3, width=1, drop_rate=0.3):
        """Constructor

        Args:
            in_planes (int): Number of channels in the input images
                Default: 3 (RGB images)
            width (int): width of the resnet
                Default: 1 (No width <=> PreActResNet38)
            drop_date (float): Dropout rate in wide basic block
        """
        super().__init__(
            PreActBlock,
            list(zip([1, 6, 6, 6], [16, 16, 32, 64])),
            in_planes,
            width=width,
            drop_rate=drop_rate,
            small_images=True,
        )
