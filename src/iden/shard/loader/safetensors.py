r"""Contain the pickle shard loader implementations."""

from __future__ import annotations

__all__ = ["TorchSafetensorsShardLoader"]

from unittest.mock import Mock

from coola.utils import is_torch_available

from iden.shard.loader.base import BaseShardLoader
from iden.shard.safetensors import TorchSafetensorsShard

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


class TorchSafetensorsShardLoader(BaseShardLoader[dict[str, torch.Tensor]]):
    r"""Implement a safetensors shard loader for ``torch.Tensor``s."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, uri: str) -> TorchSafetensorsShard:
        return TorchSafetensorsShard.from_uri(uri)
