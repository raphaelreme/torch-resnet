"""Different shortcut options."""

import sys

import torch
from torch import nn

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


__all__ = ["ConvolutionShortcut", "FullProjectionShortcut", "IdentityShortcut", "ProjectionShortcut"]


class IdentityShortcut(nn.Module):
    """Original identity shortcut from [1].

    Referenced as A in the paper:
        Identity if dimension matches
        Otherwise does simple 2d pooling with padding to match features

    Only works when in_planes <= out_planes
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.requires_activation = False

        if self.out_planes >= self.in_planes:
            raise ValueError("Identity shortcut does not handle features reduction")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:  # Simple pooling
            x = x[..., :: self.stride, :: self.stride]

        if self.out_planes > self.in_planes:  # Pad features by 0
            return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.out_planes - self.in_planes))

        return x


class ProjectionShortcut(nn.Module):
    """Original projection shortcut from [1].

    Referenced as B in the paper:
        Identity if dimension matches
        Otherwise project (and pool) features to the right dimension with a 1x1 conv
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
        super().__init__()
        self.requires_activation = stride != 1 or in_planes != out_planes

        if self.requires_activation:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.requires_activation:
            return self.conv(x)

        return x


class FullProjectionShortcut(nn.Module):
    """Original projection shortcut at each block from [1].

    Referenced as C in the paper:
        Use a 1x1 conv to project the features at each block.
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
        super().__init__()
        self.requires_activation = True

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvolutionShortcut(nn.Module):
    """Convolutional Shortcut (Similar to projection shortcut but with a 3x3 filter).

    The potential drawback of ProjectionShortcut is that it does not use half of the features when stride is 2
    (and it's even worst for greater strides) as the 1x1 filter does not take the neighbors into account.

    When dimension does not match, a 3x3 conv is applied, allowing all the information to flow.
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
        super().__init__()
        self.requires_activation = stride != 1 or in_planes != out_planes

        if self.requires_activation:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.requires_activation:
            return self.conv(x)

        return x


# ruff: disable[ERA001]
# Still have to think about it
# class IdentityShortcutV2(nn.Module):
#     """Identity shortcut but keeping all the features

#     The drawbacks of IdentityShortcut is that is uses half of the features (with stride = 2)
#     and creates half of new features initialized at 0.

#     We solve both problem by pooling the features by concatenation (only when dimension does not match)
#     """

#     def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
#         super().__init__()
#         self.requires_activation = False
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.stride = stride

#         assert self.out_planes >= self.in_planes, "Identity shortcut does not handle features reduction"

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.stride > 1:  # Simple pooling
#             x = x[..., :: self.stride, :: self.stride]

#         if self.out_planes > self.in_planes:  # Pad features by 0
#             return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.out_planes - self.in_planes))

#         return x
# ruff: enable[ERA001]
