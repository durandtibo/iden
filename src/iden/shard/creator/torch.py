r"""Contain torch shard creator implementations."""

from __future__ import annotations

__all__ = ["TorchShardCreator"]

from typing import TYPE_CHECKING, TypeVar

from coola.utils.imports import check_torch

from iden.shard import TorchShard, create_torch_shard
from iden.shard.creator.file import BaseFileShardCreator

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


class TorchShardCreator(BaseFileShardCreator[T]):
    r"""Implement a torch shard creator.

    Args:
        data: The data to save in the shard.
        path_uri: The path where to save the URI file.
        path_shard: The path where to save the shard data.
    """

    def __init__(self, data: T, path_uri: Path, path_shard: Path) -> None:
        check_torch()
        super().__init__(data=data, path_uri=path_uri, path_shard=path_shard)

    def _create(self, data: T, shard_id: str) -> TorchShard[T]:
        return create_torch_shard(
            data=data,
            uri=self._path_uri.joinpath(shard_id).as_uri(),
            path=self._path_shard.joinpath(shard_id).with_suffix(".pt"),
        )
