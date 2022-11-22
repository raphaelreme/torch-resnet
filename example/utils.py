import atexit
import pathlib
import random
from typing import List, TextIO
import sys

import numpy as np
import torch


def enforce_all_seeds(seed, strict=True):
    """Enforce all the seeds

    If strict you may have to define the following env variable:
        CUBLAS_WORKSPACE_CONFIG=:4096:8  (Increase a bit the memory foot print ~25Mo)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict:
        torch.backends.cudnn.benchmark = False  # By default should already be to False
        torch.use_deterministic_algorithms(True)


def create_seed_worker(seed, strict=True):
    """Create a callable that will seed the workers

    If used with a train data loader with random data augmentation, one should probably
    set the `persistent_workers` argument. (So that the random augmentations differs between epochs)
    """

    def seed_worker(worker_id):
        enforce_all_seeds(seed + worker_id, strict)

    return seed_worker


class StdMultiplexer:
    """Patch a writable text stream and multiplexes the outputs to several others

    Only write and flush are redirected. Other actions are done only on the main stream.
    It will therefore have the same properties as the main stream.
    """

    def __init__(self, main_stream: TextIO, ios: List[TextIO]):
        self.main_stream = main_stream
        self.ios = ios

    def write(self, string: str) -> int:
        """Write to all the streams"""
        ret = self.main_stream.write(string)

        for io_ in self.ios:
            io_.write(string)

        return ret

    def flush(self) -> None:
        """Flush all the streams"""
        self.main_stream.flush()

        for io_ in self.ios:
            io_.flush()

    def __getattr__(self, attr: str):
        return getattr(self.main_stream, attr)


class StdFileRedirection:
    """Multiplexes stdout and stderr to a file

    The code could potentially break other libraries trying to redirects sys.stdout and sys.stderr.
    It has been made compatible with Neptune. Any improvements are welcome.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.file = open(path, "w", encoding="utf-8")  # pylint: disable=consider-using-with
        self.stdout = StdMultiplexer(sys.stdout, [self.file])
        self.stderr = StdMultiplexer(sys.stderr, [self.file])
        sys.stdout = self.stdout  # type: ignore
        sys.stderr = self.stderr  # type: ignore
        atexit.register(self.close)

    def close(self):
        """Close the std file redirection (Reset sys.stdout/sys.stderr)"""
        sys.stdout = self.stdout.main_stream
        sys.stderr = self.stderr.main_stream
        self.file.close()
