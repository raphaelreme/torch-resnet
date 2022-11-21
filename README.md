# torch-resnet

Unified torch implementation of Resnets with or without pre-activation/width.

This implementation propose Resnets both for small and "large" images (Cifar vs ImageNet)
and implements all the model used in the papers introducing the ResNets.
Additional models can easily be created using the default class ResNet or PreActResNet.
It is also possible to create your own block following the same model as those implemented.

## Install

```bash
$ pip install torch-resnet
```


## Getting started

```python
import torch

import torch_resnet
from torch_resnet.utils import count_layer

model = torch_resnet.PreActResNet50()  # Build a backbone Resnet50 with pre-activation
model.set_head(nn.Linear(model.out_planes, 10))  # Set a final linear head

count_layers(model)  # -> 54 (In the original paper they do not count shortcut/downsampling layers)

out = model(torch.randn(1, 3, 224, 224))
```

## Results

Work in progress

## References

* [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. https://arxiv.org/pdf/1512.03385
* [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. https://arxiv.org/pdf/1603.05027
* [3] Sergey Zagoruyko, Nikos Komodakis
    Wide Residual Networks. https://arxiv.org/pdf/1605.07146

## Build and Deploy

```bash
$ python -m build
$ python -m twine upload dist/*
```
