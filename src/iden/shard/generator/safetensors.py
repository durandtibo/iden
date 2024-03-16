r"""Contain safetensors shard generator implementations."""

from __future__ import annotations

__all__ = ["TorchSafetensorsShardGenerator"]

from typing import TYPE_CHECKING, TypeVar
from unittest.mock import Mock

from coola.utils.imports import check_torch, is_torch_available

from iden.shard import TorchSafetensorsShard, create_torch_safetensors_shard
from iden.shard.generator.file import BaseFileShardGenerator
from iden.utils.imports import check_safetensors

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


class TorchSafetensorsShardGenerator(BaseFileShardGenerator[dict[str, torch.Tensor]]):
    r"""Implement a safetensors shard generator.

    Args:
        data: The data to save in the shard.
        path_uri: The path where to save the URI file.
        path_shard: The path where to save the shard data.
    """

    def __init__(self, data: dict[str, torch.Tensor], path_uri: Path, path_shard: Path) -> None:
        check_safetensors()
        check_torch()
        super().__init__(data=data, path_uri=path_uri, path_shard=path_shard)

    def _create(self, data: dict[str, torch.Tensor], shard_id: str) -> TorchSafetensorsShard:
        return create_torch_safetensors_shard(
            data=data,
            uri=self._path_uri.joinpath(shard_id).as_uri(),
            path=self._path_shard.joinpath(shard_id).with_suffix(".safetensors"),
        )
