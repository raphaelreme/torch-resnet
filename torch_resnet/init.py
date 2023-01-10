from __future__ import annotations

from typing import Union

import torch

from . import resnet, preact_resnet


def resnet_init(model: Union[resnet.ResNet, preact_resnet.PreActResNet], zero_init_residual=False):
    """Initialize a resnet following the paper [1]

    (More Precisely: Delving deep into rectifiers: Surpassing human-level performance on ImageNet
    classification - He, K. et al. (2015))

    Args:
        model (ResNet|PreActResNet): The resnet model to initialize
        zero_init_residual (bool): Following https://arxiv.org/abs/1706.02677 and torchvision.
            Set the weights of the final norm of each residual branch at 0 so that each block behaves
            at first as an identity. For PreActResNet, we extend this principle and set the weights of
            the final convolution of each residual branch to 0. (See Results in README.md)
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)

    if zero_init_residual:
        for module in model.modules():
            if isinstance(module, resnet.BasicBlock):
                torch.nn.init.constant_(module.bn2.weight, 0)
            elif isinstance(module, resnet.Bottleneck):
                torch.nn.init.constant_(module.bn3.weight, 0)
            elif isinstance(module, preact_resnet.PreActBlock):
                torch.nn.init.constant_(module.conv2.weight, 0)
            elif isinstance(module, preact_resnet.PreActBottleneck):
                torch.nn.init.constant_(module.conv3.weight, 0)
