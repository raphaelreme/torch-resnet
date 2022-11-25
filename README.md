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
from torch_resnet.utils import count_layer, count_parameters

model = torch_resnet.PreActResNet50()  # Build a backbone Resnet50 with pre-activation
model.set_head(nn.Linear(model.out_planes, 10))  # Set a final linear head

count_layers(model)  # -> 54 (In the original paper they do not count shortcut/downsampling layers)
count_parameters(model) / 10**6  # Nb. parameters in millions

out = model(torch.randn(1, 3, 224, 224))
```

See `example/example.py` for a more complete example.

## Results

Results obtained with `example/example.py` following the papers indication. Most are reported as mean $\pm$ std on 5 runs with different seed. If a single number is reported, only a single run has been done due to computational time.

|Model                |Params|Cifar10            |Cifar100|Cifar10 (paper)    |Cifar100 (paper)     |
|:-                   |:---- |:-----             |:-----  |:-----             |:----                |
|ResNet20             |0.3M  |8.64 $\pm$ 0.16    |TODO    |8.75 [1]           | xxx                 |
|ResNet32             |0.5M  |7.64 $\pm$ 0.23    |TODO    |7.51 [1]           | xxx                 |
|ResNet44             |0.7M  |7.47 $\pm$ 0.19    |TODO    |7.17 [1]           | xxx                 |
|ResNet56             |0.9M  |7.04 $\pm$ 0.26    |TODO    |6.97 [1]           | xxx                 |
|ResNet110            |1.7M  |6.60 $\pm$ 0.09    |TODO    |6.61 $\pm$ 0.16 [1]| xxx                 |
|ResNet164            |1.7M  |**5.97** $\pm$ 0.20|TODO    |xxx                | 25.16 [2]           |
|ResNet1001           |xxx   |TODO               |TODO    |xxx                | 27.82 [2]           |
|ResNet1202           |19.4M |7.90               |TODO    |7.93 [1]           | xxx                 |
|PreActResNet20       |0.3M  |8.61 $\pm$ 0.23    |TODO    |xxx                | xxx                 |
|PreActResNet32       |0.5M  |7.76 $\pm$ 0.10    |TODO    |xxx                | xxx                 |
|PreActResNet44       |0.7M  |7.63 $\pm$ 0.10    |TODO    |xxx                | xxx                 |
|PreActResNet56       |0.9M  |7.42 $\pm$ 0.13    |TODO    |xxx                | xxx                 |
|PreActResNet110      |1.7M  |6.79 $\pm$ 0.12    |TODO    |xxx                | xxx                 |
|PreActResNet164      |1.7M  |5.61 $\pm$ 0.16    |TODO    |5.46 [2]           | 24.33 [2]           |
|PreActResNet1001     |10.3M |4.92               |TODO    |4.89 $\pm$ 0.14 [2]| 22.68 $\pm$ 0.22 [2]|
|PreActResNet1202     |xxx   |TODO               |TODO    |xxx                | xxx                 |
|ResNet18-small       |11.2M |5.88 $\pm$ 0.15    |TODO    |xxx                | xxx                 |
|ResNet34-small       |21.3M |5.50 $\pm$ 0.17    |TODO    |xxx                | xxx                 |
|ResNet50-small       |23.5M |5.86 $\pm$ 0.30    |TODO    |xxx                | xxx                 |
|ResNet101-small      |42.5M |**5.45** $\pm$ 0.14|TODO    |xxx                | xxx                 |
|PreActResNet18-small |11.2M |5.65 $\pm$ 0.12    |TODO    |xxx                | xxx                 |
|PreActResNet34-small |21.3M |5.29 $\pm$ 0.17    |TODO    |xxx                | xxx                 |
|PreActResNet50-small |23.5M |5.83 $\pm$ 0.47    |TODO    |xxx                | xxx                 |
|PreActResNet101-small|42.5M |**5.18** $\pm$ 0.11|TODO    |xxx                | xxx                 |


TODO: Publish the training curves

## Analysis

TODO

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
