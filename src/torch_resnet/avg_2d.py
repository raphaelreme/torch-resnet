import sys

import torch
from torch import nn

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class Avg2d(nn.Module):
    """Equivalent to AdaptiveAveragePooling(1) but simpler and squeeze spatial dimension."""

    @override
    def forward(self, x):
        return torch.mean(x, dim=(-1, -2))
