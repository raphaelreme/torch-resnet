import torch
from torch import nn


class Avg2d(nn.Module):
    """Equivalent to AdaptiveAveragePooling(1) but simpler and squeeze spatial dimension"""

    def forward(self, x):
        return torch.mean(x, dim=(-1, -2))
