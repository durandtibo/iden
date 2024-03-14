r"""Contain JSON shard creator implementations."""

from __future__ import annotations

__all__ = ["JsonShardCreator"]

from typing import TypeVar

from iden.shard import JsonShard, create_json_shard
from iden.shard.creator.file import BaseFileShardCreator

T = TypeVar("T")


class JsonShardCreator(BaseFileShardCreator[T]):
    r"""Implement a JSON shard creator.

    Args:
        data: The data to save in the shard.
        path_uri: The path where to save the URI file.
        path_shard: The path where to save the shard data.
    """

    def _create(self, data: T, shard_id: str) -> JsonShard[T]:
        return create_json_shard(
            data=data,
            uri=self._path_uri.joinpath(shard_id).as_uri(),
            path=self._path_shard.joinpath(shard_id).with_suffix(".json"),
        )
