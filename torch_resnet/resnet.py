"""Classical resnets [1]

Note that in [3], Wide ResNet 50-2 is without pre-activation contrary to Wide ResNet16/28/40
"""

from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from . import init
from .avg_2d import Avg2d
from .shortcut import ProjectionShortcut


__all__ = [
    "ResNet",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNet200",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "ResNet56",
    "ResNet110",
    "ResNet164",
    "ResNet1001",
    "ResNet1202",
    "WideResNet16",
    "WideResNet28",
    "WideResNet40",
]


class BasicBlock(nn.Module):
    """Basic bloc"""

    def __init__(
        self,
        in_planes: int,
        planes: int,
        *,
        shortcut: type = ProjectionShortcut,
        stride=1,
        width=1,
        drop_rate=0.0,
    ):
        """Constructor

        Args:
            in_planes (int): Input dimension of features
            planes (int): Default target dimension of features (before applying width or expansion)
                See self.out_planes to know the real output dimension
            shortcut (type): Constructor for shortcut blocks
                Default: ProjectionShortcut
            stride (int): Stride of the first convolution to downsample spatial dimensions
                usally set to 1 (no downsampling) or 2 (downsample by 2)
            width (int): Width of the block for WideResNet.
            drop_rate (float): dropout rate to apply between the convolution.
        """
        super().__init__()
        self.out_planes = planes * width

        self.conv1 = nn.Conv2d(in_planes, self.out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_planes)
        self.dropout = nn.Dropout(drop_rate, inplace=True)
        self.conv2 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_planes)

        self.shortcut = nn.Sequential(shortcut(in_planes, self.out_planes, stride))
        if self.shortcut[0].requires_activation:
            self.shortcut.insert(1, nn.BatchNorm2d(self.out_planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(self.dropout(out)))  # Dropout after relu, and before second conv cf [3]

        out += self.shortcut(x)

        return F.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck block

    Expand features dimension by expansion
    """

    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        *,
        shortcut: type = ProjectionShortcut,
        stride=1,
        width=1,
        drop_rate=0.0,
    ):
        """Constructor

        Args:
            in_planes (int): Input dimension of features
            planes (int): Default target dimension of features (before applying width or expansion)
                See self.out_planes to know the real output dimension
            shortcut (type): Constructor for shortcut blocks
                Default: ProjectionShortcut
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

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, self.out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_planes)

        self.shortcut = nn.Sequential(shortcut(in_planes, self.out_planes, stride))
        if self.shortcut[0].requires_activation:
            self.shortcut.insert(1, nn.BatchNorm2d(self.out_planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)

        return F.relu(out)


class ResNet(nn.Module):
    """Base class for Resnet"""

    def __init__(
        self,
        dimensions: List[Tuple[int, int]],
        block: type = BasicBlock,
        *,
        shortcut: type = ProjectionShortcut,
        in_planes=3,
        width=1,
        drop_rate=0.0,
        small_images=False,
        zero_init_residual=False,
    ):
        """Constructor

        Args:
            dimensions (List[Tuple[int, int]]): All the dimensions of the resnet as (num_blocks, planes)
                for each main layer.
            block (type): Constructor of a block (BasicBlock or Bottleneck or your own)
                Default: BasicBlock
            shortcut (type): Constructor for shortcut blocks
                Default: ProjectionShortcut
            in_planes (int): Number of channels in the input images
                Default: 3 (standard images)
            width (int): Width of Wide-Resnet
                Default: 1 (Standard Resnet)
            drop_rate (float): Dropout rate for Wide-Resnet with basic block
                Default: 0.0
            small_images (bool): With small images (size < 100px), let's rather use Cifar version of ResNet
                This results in changing the first 7x7 conv by a 3x3 conv without stride and max pooling
                Default: False
            zero_init_residual (bool): Initialize the residual branch last bn weight at 0 (See `resnet_init`)
                Default: False
        """
        super().__init__()

        self.out_planes = dimensions[0][1]

        self.init_layer: nn.Module
        if small_images:
            self.init_layer = nn.Sequential(
                nn.Conv2d(in_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.out_planes),
                nn.ReLU(inplace=True),
            )
        else:
            self.init_layer = nn.Sequential(
                nn.Conv2d(in_planes, self.out_planes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.out_planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        strides = [1] + [2] * (len(dimensions) - 2)
        layers: List[nn.Module] = []
        for (num_blocks, planes), stride in zip(dimensions[1:], strides):
            layers.append(
                self._make_layer(
                    block=block,
                    shortcut=shortcut,
                    planes=planes,
                    num_blocks=num_blocks,
                    stride=stride,
                    width=width,
                    drop_rate=drop_rate,
                )
            )

        self.layers = nn.Sequential(*layers)
        self.avgpool = Avg2d()
        self.head: nn.Module = nn.Identity()

        init.resnet_init(self, zero_init_residual)

    def _make_layer(
        self, *, block: type, shortcut: type, planes: int, num_blocks: int, stride: int, width: int, drop_rate: float
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_ in strides:
            layers.append(
                block(self.out_planes, planes, shortcut=shortcut, stride=stride_, width=width, drop_rate=drop_rate)
            )
            self.out_planes = layers[-1].out_planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.init_layer(x)
        out = self.layers(out)
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


class ResNet18(ResNet):
    """Resnet 18

    Was designed for ImageNet but can be adapted to cifar with small_images parameter
    """

    def __init__(self, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(list(zip([1, 2, 2, 2, 2], [64, 64, 128, 256, 512])), BasicBlock, **kwargs)


class ResNet34(ResNet):
    """Resnet 34

    Was designed for ImageNet but can be adapted to cifar with small_images parameter
    """

    def __init__(self, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(list(zip([1, 3, 4, 6, 3], [64, 64, 128, 256, 512])), BasicBlock, **kwargs)


class ResNet50(ResNet):
    """Resnet 50

    Was designed for ImageNet but can be adapted to Cifar with small_images parameter

    WideResNet50-2 can easily be build by passing width=2
    """

    def __init__(self, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(list(zip([1, 3, 4, 6, 3], [64, 64, 128, 256, 512])), Bottleneck, **kwargs)


class ResNet101(ResNet):
    """Resnet 101

    Was designed for ImageNet but can be adapted to Cifar with small_images parameter

    WideResNet101-2 can easily be build by passing width=2
    """

    def __init__(self, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(list(zip([1, 3, 4, 23, 3], [64, 64, 128, 256, 512])), Bottleneck, **kwargs)


class ResNet152(ResNet):
    """Resnet 152

    Was designed for ImageNet but can be adapted to Cifar with small_images parameter
    """

    def __init__(self, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(list(zip([1, 3, 8, 36, 3], [64, 64, 128, 256, 512])), Bottleneck, **kwargs)


class ResNet200(ResNet):
    """Resnet 200

    Was designed for ImageNet but can be adapted to Cifar with small_images parameter
    """

    def __init__(self, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(list(zip([1, 3, 24, 36, 3], [64, 64, 128, 256, 512])), Bottleneck, **kwargs)


# Designed for Cifar


class ResNet20(ResNet):
    """Resnet 20

    Was designed for Cifar.
    """

    def __init__(self, small_images=True, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(list(zip([1, 3, 3, 3], [16, 16, 32, 64])), BasicBlock, small_images=small_images, **kwargs)


class ResNet32(ResNet):
    """Resnet 32

    Was designed for Cifar.
    """

    def __init__(self, small_images=True, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(list(zip([1, 5, 5, 5], [16, 16, 32, 64])), BasicBlock, small_images=small_images, **kwargs)


class ResNet44(ResNet):
    """Resnet 44

    Was designed for Cifar.
    """

    def __init__(self, small_images=True, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(list(zip([1, 7, 7, 7], [16, 16, 32, 64])), BasicBlock, small_images=small_images, **kwargs)


class ResNet56(ResNet):
    """Resnet 56

    Was designed for Cifar.
    """

    def __init__(self, small_images=True, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(list(zip([1, 9, 9, 9], [16, 16, 32, 64])), BasicBlock, small_images=small_images, **kwargs)


class ResNet110(ResNet):
    """Resnet 110

    Was designed for Cifar.
    """

    def __init__(self, small_images=True, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(list(zip([1, 18, 18, 18], [16, 16, 32, 64])), BasicBlock, small_images=small_images, **kwargs)


class ResNet164(ResNet):
    """Resnet 164

    Was designed for Cifar.
    """

    def __init__(self, small_images=True, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        # Following [2] -> Usage of bottle neck
        super().__init__(list(zip([1, 18, 18, 18], [16, 16, 32, 64])), Bottleneck, small_images=small_images, **kwargs)


class ResNet1001(ResNet):
    """Resnet 1001

    Was designed for Cifar.
    """

    def __init__(self, small_images=True, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        # Following [2] -> Usage of bottle neck
        super().__init__(
            list(zip([1, 111, 111, 111], [16, 16, 32, 64])), Bottleneck, small_images=small_images, **kwargs
        )


class ResNet1202(ResNet):
    """Resnet 1202

    Was designed for Cifar. Performs poorly according to [1].
    """

    def __init__(self, small_images=True, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        # Following [1] -> Default block
        super().__init__(
            list(zip([1, 200, 200, 200], [16, 16, 32, 64])), BasicBlock, small_images=small_images, **kwargs
        )


class WideResNet16(ResNet):
    """WideResnet 16-k without preactivation

    Was designed for Cifar.

    The number of layer matches the paper [3] and does not follow [1] and [2]:
    The shortcut layers (2 for these models) are added to the count. (They assume ProjectionShortcut)
    """

    def __init__(self, drop_rate=0.3, small_images=True, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(
            list(zip([1, 2, 2, 2], [16, 16, 32, 64])),
            BasicBlock,
            drop_rate=drop_rate,
            small_images=small_images,
            **kwargs,
        )


class WideResNet28(ResNet):
    """WideResnet 28-k without preactivation

    Was designed for Cifar.

    The number of layer matches the paper [3] and does not follow [1] and [2]:
    The shortcut layers (2 for these models) are added to the count. (They assume ProjectionShortcut)
    """

    def __init__(self, drop_rate=0.3, small_images=True, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(
            list(zip([1, 4, 4, 4], [16, 16, 32, 64])),
            BasicBlock,
            drop_rate=drop_rate,
            small_images=small_images,
            **kwargs,
        )


class WideResNet40(ResNet):
    """WideResnet 40-k without preactivation

    Was designed for Cifar.

    The number of layer matches the paper [3] and does not follow [1] and [2]:
    The shortcut layers (2 for these models) are added to the count. (They assume ProjectionShortcut)
    """

    def __init__(self, drop_rate=0.3, small_images=True, **kwargs):
        """Constructor

        Args:
            kwargs: Any valid argument for `ResNet` class (See the documentation)
                Except dimensions and block
        """
        super().__init__(
            list(zip([1, 6, 6, 6], [16, 16, 32, 64])),
            BasicBlock,
            drop_rate=drop_rate,
            small_images=small_images,
            **kwargs,
        )
