from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class Shortcut(Protocol):  # noqa: D101
    requires_activation: bool

    def __init__(self, in_planes: int, out_planes: int, stride: int): ...
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...  # noqa: D102
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...  # noqa: D102


@runtime_checkable
class Block(Protocol):  # noqa: D101
    out_planes: int

    def __init__(
        self,
        in_planes: int,
        planes: int,
        *,
        shortcut: type[Shortcut],
        stride: int = 1,
        width: int = 1,
        drop_rate: float = 0.0,
    ): ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...  # noqa: D102
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...  # noqa: D102
