# torch-resnet

Unified torch implementation of resnets with or without pre-activation/width.

We implement (pre-act) resnets for ImageNet for each size described in [1] and [2] (18, 34, 50, 101, 152, 200).
We also implement (pre-act) resnets for Cifar for each size described in [1] and [2] (20, 32, 44, 56, 110, 164, 1001, 1202).

Following what is used in the literature we also propose a version of the ImageNet resnets for Cifar10 (See SimClr paper https://arxiv.org/pdf/2002.05709.pdf). To adapt the architectures
to smaller images (32x32), the initial layers conv 7x7 and max_pool are replaced by a simple conv 3x3. The other layers/parameters are kept, resulting in a variant of wide-resnets for Cifar.

Finally, we implement wide resnets (with or without pre-activation) for Cifar and ImageNet following [3].

Additional models can easily be created using the default class ResNet or PreActResNet.
It is also possible to create your own block following the same model as those implemented. (To create for instance the different variations of resnets proposed in [1], [2])

We use by default projection shortcuts whenever they are required (option B [1]) but we have also implemented option A and C, and more can be added following the same template.

We have validated our implementation by testing it on Cifar10/Cifar100 (See Results).

## Install

```bash
$ pip install torch-resnet
```


## Getting started

```python
import torch

import torch_resnet
from torch_resnet.utils import count_layer, count_parameters

model = torch_resnet.PreActResNet50()  # Build a backbone Resnet50 with pre-activation for ImageNet
model.set_head(nn.Linear(model.out_planes, 1000))  # Set a final linear head

count_layers(model)  # -> 54 (In the original paper they do not count shortcut/downsampling layers)
count_parameters(model) / 10**6  # Nb. parameters in millions

out = model(torch.randn(1, 3, 224, 224))
```

See `example/example.py` for a more complete example.

## Results

Results obtained with `example/example.py` following closely papers indications. Most are reported as mean $\pm$ std (on 5 runs with different seed). If a single number is reported, only a single run has been done due to computational time. All training are done with Automatique Mixed Precision.

For all resnets and pre-act resnets, the learning rate is scheduled following [1] and [2] with a warm-up in the 400 first iterations at 0.01. The initial learning rate is then set at 0.1 and decreased by 10 at 32k and 48k
iterations. Training is stopped after 160 epochs (~62.5k iterations).

Contrary to [1], the training set is not split in 45k/5k to perform validation, and we directly took the final model to evaluate the performances on the
test set (No validation is done). Also on Cifar [1] is using option A (Identity shortcut), whereas we use the default option B (Projection shortcut when required).

Follow the link to access the training curves for each model and each seed used (111, 222, 333, 444, 555). (Upload in coming...)


|Model                |Params|Cifar10                                                                         |Cifar10 (paper)    |Cifar100                                                                      |Cifar100 (paper)    |
|:-                   |:---- |:-----                                                                          |:-----             |:-----                                                                        |:----               |
|ResNet20             |0.3M  |[8.64 $\pm$ 0.16](https://tensorboard.dev/experiment/Wp0BnqgsQAiNlda6jbHyNw)    |8.75 [1]           |33.23 $\pm$ 0.32                                                              |xxx                 |
|ResNet32             |0.5M  |[7.64 $\pm$ 0.23](https://tensorboard.dev/experiment/jS3iZpYnSXe0JWbmdmsJRg)    |7.51 [1]           |31.64 $\pm$ 0.54                                                              |xxx                 |
|ResNet44             |0.7M  |[7.47 $\pm$ 0.19](https://tensorboard.dev/experiment/Vp32WK2GTEu6EkyTdAc9gg)    |7.17 [1]           |30.88 $\pm$ 0.22                                                              |xxx                 |
|ResNet56             |0.9M  |[7.04 $\pm$ 0.26](https://tensorboard.dev/experiment/I6RV0GFqQni9LnWxRnBFWw)    |6.97 [1]           |30.15 $\pm$ 0.29                                                              |xxx                 |
|ResNet110            |1.7M  |[6.60 $\pm$ 0.09](https://tensorboard.dev/experiment/oh0WxpW0Sw6Yrq5vRBWbVg)    |6.61 $\pm$ 0.16 [1]|28.99 $\pm$ 0.22                                                              |xxx                 |
|ResNet164            |1.7M  |[**5.97** $\pm$ 0.20](https://tensorboard.dev/experiment/obTXYd9NSx2xErmzvHwzjw)|xxx                |25.79 $\pm$ 0.51                                                              |25.16 [2]           |
|ResNet1001*          |10.3M |[7.95](https://tensorboard.dev/experiment/pZCAbqaoQR2I2T9mLGZZzg)               |xxx                |29.94                                                                         |27.82 [2]           |
|ResNet1202           |19.4M |[7.90](https://tensorboard.dev/experiment/dUMT2GoDR7GgDgnun6oDLA)               |7.93 [1]           |33.20                                                                         |xxx                 |
|PreActResNet20       |0.3M  |[8.61 $\pm$ 0.23](https://tensorboard.dev/experiment/RlKrhoLnRA6ptZcns3R7fA)    |xxx                |33.40 $\pm$ 0.30                                                              |xxx                 |
|PreActResNet32       |0.5M  |[7.76 $\pm$ 0.10](https://tensorboard.dev/experiment/LcNVsEdzSBKeZJygGBTpEw)    |xxx                |xx.xx $\pm$ x.xx                                                              |xxx                 |
|PreActResNet44       |0.7M  |[7.63 $\pm$ 0.10](https://tensorboard.dev/experiment/I4yJquNxQ8eDdgKcxUDE7A)    |xxx                |30.78 $\pm$ 0.17                                                              |xxx                 |
|PreActResNet56       |0.9M  |[7.42 $\pm$ 0.13](https://tensorboard.dev/experiment/XdcNemL8Ta2WbVXWq1aTeg)    |xxx                |30.18 $\pm$ 0.39                                                              |xxx                 |
|PreActResNet110      |1.7M  |[6.79 $\pm$ 0.12](https://tensorboard.dev/experiment/TpDZKZmqTlS6NyFVL1ebAQ)    |xxx                |28.45 $\pm$ 0.25                                                              |xxx                 |
|PreActResNet164      |1.7M  |[5.61 $\pm$ 0.16](https://tensorboard.dev/experiment/L4V78FG7T2a4OYxAGnN7jQ)    |5.46 [2]           |25.23 $\pm$ 0.21                                                              |24.33 [2]           |
|PreActResNet1001     |10.3M |[**4.92**](https://tensorboard.dev/experiment/VUM0SW35Rc24wasG8S79GA)           |4.89 $\pm$ 0.14 [2]|23.18                                                                         |22.68 $\pm$ 0.22 [2]|
|PreActResNet1202     |19.4M |[6.66](https://tensorboard.dev/experiment/e7WPBDcbRwuZRJReQsoPFg)               |xxx                |27.65                                                                         |xxx                 |
|ResNet18-small       |11.2M |5.88 $\pm$ 0.15                                                                 |xxx                |26.74 $\pm$ 0.42                                                              |xxx                 |
|ResNet34-small       |21.3M |5.50 $\pm$ 0.17                                                                 |xxx                |25.34 $\pm$ 0.29                                                              |xxx                 |
|ResNet50-small       |23.5M |5.86 $\pm$ 0.30                                                                 |xxx                |25.20 $\pm$ 0.89                                                              |xxx                 |
|ResNet101-small      |42.5M |**5.45** $\pm$ 0.14                                                             |xxx                |**23.93** $\pm$ 0.56                                                          |xxx                 |
|PreActResNet18-small |11.2M |5.65 $\pm$ 0.12                                                                 |xxx                |25.46 $\pm$ 0.34                                                              |xxx                 |
|PreActResNet34-small |21.3M |5.29 $\pm$ 0.17                                                                 |xxx                |24.75 $\pm$ 0.31                                                              |xxx                 |
|PreActResNet50-small |23.5M |5.83 $\pm$ 0.47                                                                 |xxx                |23.97 $\pm$ 0.36                                                              |xxx                 |
|PreActResNet101-small|42.5M |**5.18** $\pm$ 0.11                                                             |xxx                |**23.69** $\pm$ 0.41                                                          |xxx                 |

\* ResNet1001 cannot be trained with AMP (due to training instability) thus it was trained without AMP. Also, please note that AMP usually leads to slightly worst performances, therefore most of our results here are probably underestimated.

Note that in [2] and in most github implementation, the test set is used as a validation set (taking the max acc reached on it as the final result, as done in the [official implem](https://github.com/facebookarchive/fb.resnet.torch) [2]), obviously leading to falsely better performances.  When dropping AMP and taking the max value rather than the last value, we also reach better performances. (We only tested on PreActResNet164 and PreActResNet1001).

|Model (No Amp, best)|Params|Cifar100                                                                      |Cifar100 (paper)    |
|:-                  |:---- |:-----                                                                        |:----               |
|PreActResNet164     |1.7M  |24.83 $\pm$ 0.16                                                              |24.33 [2]           |
|PreActResNet1001    |10.3M |22.86                                                                         |22.68 $\pm$ 0.22 [2]|




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
