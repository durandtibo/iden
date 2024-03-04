r"""Contain loaders to load data in a safetensors format."""

from __future__ import annotations

__all__ = ["NumpySafetensorsLoader", "TorchSafetensorsLoader"]

from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.utils import check_numpy, check_torch, is_numpy_available, is_torch_available

from iden.io.base import BaseLoader
from iden.utils.imports import check_safetensors, is_safetensors_available
from iden.utils.path import sanitize_path

if TYPE_CHECKING:
    from pathlib import Path

if is_numpy_available():
    import numpy as np
else:
    np = Mock()  # pragma: no cover

if is_safetensors_available():
    from safetensors import numpy as sn
    from safetensors import torch as st
else:  # pragma: no cover
    sn = Mock()
    st = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


class NumpySafetensorsLoader(BaseLoader[dict[str, np.ndarray]]):
    r"""Implement a file loader to load ``numpy.ndarray``s in the
    safetensors format.

    Link: https://huggingface.co/docs/safetensors/en/index
    """

    def __init__(self) -> None:
        check_safetensors()
        check_numpy()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, path: Path) -> dict[str, np.ndarray]:
        return sn.load_file(sanitize_path(path))


class TorchSafetensorsLoader(BaseLoader[dict[str, torch.Tensor]]):
    r"""Implement a file loader to load ``torch.Tensor``s in the
    safetensors format.

    Link: https://huggingface.co/docs/safetensors/en/index
    """

    def __init__(self, device: str | dict = "cpu") -> None:
        check_safetensors()
        check_torch()
        self._device = device

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(device={self._device})"

    def load(self, path: Path) -> dict[str, torch.Tensor]:
        return st.load_file(sanitize_path(path), device=self._device)
