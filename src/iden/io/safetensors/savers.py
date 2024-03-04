r"""Contain implementation of savers that use the safetensors format."""

from __future__ import annotations

__all__ = ["TorchSafetensorsSaver"]

from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.utils import check_torch, is_torch_available

from iden.io import BaseFileSaver
from iden.utils.imports import check_safetensors, is_safetensors_available

if TYPE_CHECKING:
    from pathlib import Path

if is_safetensors_available():
    from safetensors import torch as st
else:  # pragma: no cover
    st = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


class TorchSafetensorsSaver(BaseFileSaver[dict[str, torch.Tensor]]):
    r"""Implement a file saver to save ``torch.Tensor``s with the
    safetensors format.

    This saver can only save a dictionary of ``torch.Tensor``s.

    Link: https://huggingface.co/docs/safetensors/en/index
    """

    def __init__(self) -> None:
        check_safetensors()
        check_torch()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _save_file(self, to_save: dict[str, torch.Tensor], path: Path) -> None:
        st.save_file(to_save, path)
