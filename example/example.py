"""Reproducing cifar results with ResNets architectures"""

import argparse
import os
import pathlib
from typing import Union

from deep_trainer import PytorchTrainer
from deep_trainer.pytorch import metric
import torch
import torch.utils.data
import torchvision  # type: ignore
import yaml

import torch_resnet
from torch_resnet.utils import count_layers, count_parameters

from utils import create_seed_worker, enforce_all_seeds, StdFileRedirection


MEANS = {
    "CIFAR10": (0.4914, 0.4822, 0.4465),
    "CIFAR100": (0.5071, 0.4867, 0.4408),
}

STDS = {
    "CIFAR10": (0.2023, 0.1994, 0.2010),
    "CIFAR100": (0.2675, 0.2565, 0.2761),
}


def parse_model_name(model_name: str) -> Union[torch_resnet.ResNet, torch_resnet.PreActResNet]:
    """Parse model name

    Format: ClassName[-small][-width] (small and width are exclusive)

    Able to parse WideResNet40-10 or ResNet50-small
    """
    model_name, *specs = model_name.split("-")
    if len(specs) > 1:
        raise ValueError(f"Too many specifications for {model_name}: {specs}")

    kwargs: dict = {}
    if specs:
        if specs[0] == "small":
            kwargs["small_images"] = True
        else:
            kwargs["width"] = int(specs[0])

    model: Union[torch_resnet.ResNet, torch_resnet.PreActResNet]
    try:
        model = getattr(torch_resnet, model_name)(**kwargs)
    except AttributeError as exc:
        raise ValueError(f"Invalid model name {model_name}") from exc

    return model


def build_resnet_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """Build the scheduler used in [1] and [2]"""

    def _lambda(step):
        if step <= 400:
            return 0.1

        if step <= 32000:
            return 1.0

        if step <= 48000:
            return 0.1

        return 0.01

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lambda)


def main(
    model_name: str,
    dataset: str,
    checkpoint: str,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    use_amp: bool,
    eval_best: bool,
    seed: int,
    override_optim_hp: bool,
):
    enforce_all_seeds(seed)

    # Data
    assert dataset in MEANS, "Dataset supported are CIFAR10 and CIFAR100"

    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(MEANS[dataset], STDS[dataset]),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(MEANS[dataset], STDS[dataset]),
        ]
    )

    trainset = getattr(torchvision.datasets, dataset)(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=create_seed_worker(seed),
        persistent_workers=True,
    )

    testset = getattr(torchvision.datasets, dataset)(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=8,
        worker_init_fn=create_seed_worker(seed),
        persistent_workers=True,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    # Model
    model = parse_model_name(model_name)
    model.set_head(torch.nn.Linear(model.out_planes, len(trainset.classes)))
    model.to(device)
    print(model)

    # Optimizer and Scheduler
    no_decay = []  # Apply weight_decay only on weights of conv and linear
    decay = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            decay.append(module.weight)
            if module.bias is not None:
                no_decay.append(module.bias)
        else:
            no_decay.extend(module.parameters(False))

    optimizer = torch.optim.SGD(
        [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}],
        lr=lr,
        momentum=0.9,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainset) / batch_size * epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [32000, 48000], 0.1)
    scheduler = build_resnet_scheduler(optimizer)

    # Criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Metrics
    metrics_handler = metric.MetricsHandler(
        [
            metric.Accuracy(),
        ]
    )

    # Trainer
    trainer = PytorchTrainer(
        model,
        optimizer,
        scheduler,
        metrics_handler,
        device,
        output_dir=f"experiments/{dataset}/{model_name}/{seed}",
        save_mode="small",
        use_amp=use_amp,
    )

    if checkpoint:
        print(f"Reload from {checkpoint}")
        trainer.load(checkpoint, restore_optim_hp=not override_optim_hp)
        print("Compute metrics with the loaded checkpoint", flush=True)
        print(trainer.evaluate(test_loader))

    # Train
    print("Training....")
    trainer.train(epochs, train_loader, criterion, test_loader)

    if eval_best:
        print("Loading best model")
        trainer.load(f"{trainer.output_dir}/checkpoints/best.ckpt")

    # Evaluate
    print("Evaluate...")
    metrics = trainer.evaluate(test_loader)
    metrics["num_layers"] = count_layers(model)
    metrics["num_parameters"] = count_parameters(model) / 10**6

    print(yaml.dump(metrics))

    with open(f"experiments/{dataset}/{model_name}/{seed}/metrics.yml", "w") as file:
        file.write(yaml.dump(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a resnet on cifar10/100")
    parser.add_argument("--model", default="ResNet20", help="Model name")
    parser.add_argument("--data", default="CIFAR10", help="Dataset [CIFAR10/CIFAR100]")
    parser.add_argument("--ep", default=160, type=int, help="epochs")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--bs", default=128, type=int, help="batch size")
    parser.add_argument("--wd", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--no-amp", action="store_true", help="Do not use AMP")
    parser.add_argument("--best", action="store_true", help="Eval best rather than last")
    parser.add_argument("--seed", default=666, type=int, help="Seed")
    parser.add_argument("--ckpt", help="Checkpoint to restore")
    parser.add_argument(
        "--override-optim-hp",
        action="store_true",
        help="Override optimizer and scheduler hyper parameters from checkpoint",
    )

    args = parser.parse_args()

    exp_path = pathlib.Path("experiments") / f"{args.data}" / f"{args.model}" / f"{args.seed}"
    exp_path.mkdir(parents=True)
    StdFileRedirection(exp_path / "outputs.log")

    print(args)
    main(
        args.model,
        args.data,
        args.ckpt,
        args.ep,
        args.lr,
        args.bs,
        args.wd,
        not args.no_amp,
        args.best,
        args.seed,
        args.override_optim_hp,
    )
