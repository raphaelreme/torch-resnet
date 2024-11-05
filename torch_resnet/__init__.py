"""Resnets in PyTorch.

Implementation follows the papers [1], [2] and [3] to build classical resnets, pre-activation resnets and wide resnets.

Bottleneck follows torchvision and nvidia with the stride on the 3x3 convolution (Resnet v1.5)
https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. https://arxiv.org/pdf/1512.03385
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. https://arxiv.org/pdf/1603.05027
[3] Sergey Zagoruyko, Nikos Komodakis
    Wide Residual Networks. https://arxiv.org/pdf/1605.07146
"""

from .preact_resnet import *
from .resnet import *
from .shortcut import *


__version__ = "0.0.4"
