from typing import Tuple

from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Count the total of learnable parameters of the model

    Args:
        model (nn.Module)

    Returns:
        int: Number of learnable parameters in model
    """
    count = 0
    for parameters in model.parameters():
        if parameters.requires_grad:
            count += parameters.numel()

    return count


def count_layers(model: nn.Module, layer_type: Tuple[type, ...] = (nn.Conv2d, nn.Linear)) -> int:
    """Count the number of layer matching layer_type

    Args:
        model (nn.Module)
        layer_type (Tuple[type, ...]): Types of layer to count

    Returns:
        int: Number of layer in model
    """
    count = 0
    for module in model.modules():
        if isinstance(module, layer_type):
            count += 1

    return count
