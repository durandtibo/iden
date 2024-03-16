r"""Contain tuple-based shard generator implementations."""

from __future__ import annotations

__all__ = ["ShardTupleGenerator"]

from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping

from iden.shard import BaseShard, ShardTuple, create_shard_tuple
from iden.shard.generator.base import BaseShardGenerator, setup_shard_generator

if TYPE_CHECKING:
    from pathlib import Path


class ShardTupleGenerator(BaseShardGenerator[tuple[BaseShard, ...]]):
    r"""Implement a ``ShardTuple`` generator.

    Args:
        shard: The shard generator or its configuration.
        num_shards: The number of shards to generate in the
            ``ShardTuple``.
        path_uri: The path where to save the URI file.
    """

    def __init__(self, shard: BaseShardGenerator | dict, num_shards: int, path_uri: Path) -> None:
        self._shard = setup_shard_generator(shard)
        self._num_shards = num_shards
        self._path_uri = path_uri

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "path_uri": self._path_uri,
                    "num_shards": self._num_shards,
                    "shard": self._shard,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, shard_id: str) -> ShardTuple:
        shards = [self._shard.generate(f"{i + 1:09}") for i in range(self._num_shards)]
        return create_shard_tuple(uri=self._path_uri.joinpath(shard_id).as_uri(), shards=shards)
