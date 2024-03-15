r"""Contain tuple-based shard creator implementations."""

from __future__ import annotations

__all__ = ["ShardTupleCreator"]

from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping

from iden.shard import BaseShard, ShardTuple
from iden.shard.creator.base import BaseShardCreator

if TYPE_CHECKING:
    from pathlib import Path


class ShardTupleCreator(BaseShardCreator[tuple[BaseShard, ...]]):
    r"""Implement a ``ShardTuple`` creator.

    Args:
        shard: The shard creator or its configuration.
        num_shards: The number of shards to generate in the
            ``ShardTuple``.
        path_uri: The path where to save the URI file.
    """

    def __init__(self, shard: BaseShardCreator, num_shards: int, path_uri: Path) -> None:
        self._shard = shard
        self._num_shards = num_shards
        self._path_uri = path_uri

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "shard": self._shard,
                    "num_shards": self._num_shards,
                    "path_uri": self._path_uri,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def create(self, shard_id: str) -> ShardTuple:
        shards = [self._shard.create(f"{i+1:09}") for i in range(self._num_shards)]
        return ShardTuple(uri=self._path_uri.joinpath(shard_id).as_uri(), shards=shards)
