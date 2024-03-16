r"""Contain JSON shard generator implementations."""

from __future__ import annotations

__all__ = ["JsonShardGenerator"]

from typing import TypeVar

from iden.shard import JsonShard, create_json_shard
from iden.shard.generator.file import BaseFileShardGenerator

T = TypeVar("T")


class JsonShardGenerator(BaseFileShardGenerator[T]):
    r"""Implement a JSON shard generator.

    Args:
        data: The data to save in the shard.
        path_uri: The path where to save the URI file.
        path_shard: The path where to save the shard data.
    """

    def _generate(self, data: T, shard_id: str) -> JsonShard[T]:
        return create_json_shard(
            data=data,
            uri=self._path_uri.joinpath(shard_id).as_uri(),
            path=self._path_shard.joinpath(shard_id).with_suffix(".json"),
        )
