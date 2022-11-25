import argparse
import pathlib
import re
from typing import Iterable, List, Union

import numpy as np
import yaml


def sorted_alphanumeric(data: Iterable[pathlib.Path]) -> List[pathlib.Path]:
    """Sorts alphanumeriacally an iterable of path

    "1" < "2" < "10" < "foo1" < "foo2" < "foo3"
    """

    def convert(text: str) -> Union[int, str]:
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key: pathlib.Path) -> List[Union[int, str]]:
        return [convert(c) for c in re.split("([0-9]+)", key.name)]

    return sorted(data, key=alphanum_key)


def main(data_dir: str):
    root = pathlib.Path(data_dir)
    assert root.is_dir()

    print(f"{'model':22}|{'mean':10}|{'std':10}|{'min':10}|{'max':10}|{'Params':11}|{'n_runs':8}")

    for model_path in sorted_alphanumeric(list(root.iterdir())):
        if not model_path.is_dir():
            continue
        model_name = model_path.name
        errors = []

        for exp_path in model_path.iterdir():
            metrics_path = exp_path / "metrics.yml"
            if not metrics_path.exists():
                continue

            metrics = yaml.safe_load((exp_path / "metrics.yml").read_text())
            errors.append(100 - 100 * metrics["Accuracy"])

        if errors:
            print(
                f"{model_name:22}|{round(np.mean(errors), 2):10}|{round(np.std(errors), 2):10}|{round(min(errors), 2):10}|{round(max(errors), 2):10}|{round(metrics['num_parameters'], 2):10}M|{len(errors):8}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate all the results")
    parser.add_argument("--data-dir", default="experiments/CIFAR10", help="Base path to the main directory")
    args = parser.parse_args()

    main(args.data_dir)
