r"""Contain pickle shard generator implementations."""

from __future__ import annotations

__all__ = ["PickleShardGenerator"]

from typing import TypeVar

from iden.shard import PickleShard, create_pickle_shard
from iden.shard.generator.file import BaseFileShardGenerator

T = TypeVar("T")


class PickleShardGenerator(BaseFileShardGenerator[T]):
    r"""Implement a pickle shard generator.

    Args:
        data: The data to save in the shard.
        path_uri: The path where to save the URI file.
        path_shard: The path where to save the shard data.
    """

    def _create(self, data: T, shard_id: str) -> PickleShard[T]:
        return create_pickle_shard(
            data=data,
            uri=self._path_uri.joinpath(shard_id).as_uri(),
            path=self._path_shard.joinpath(shard_id).with_suffix(".pkl"),
        )
