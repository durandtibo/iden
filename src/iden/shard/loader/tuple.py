r"""Contain shard loader implementations."""

from __future__ import annotations

__all__ = ["ShardTupleLoader"]


from iden.shard.base import BaseShard
from iden.shard.loader.base import BaseShardLoader
from iden.shard.tuple import ShardTuple


class ShardTupleLoader(BaseShardLoader[tuple[BaseShard, ...]]):
    r"""Implement a ``ShardTuple`` loader."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, uri: str) -> ShardTuple:
        return ShardTuple.from_uri(uri)