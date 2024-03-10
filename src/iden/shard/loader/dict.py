r"""Contain shard loader implementations."""

from __future__ import annotations

__all__ = ["ShardDictLoader"]


from iden.shard.base import BaseShard
from iden.shard.dict import ShardDict
from iden.shard.loader.base import BaseShardLoader


class ShardDictLoader(BaseShardLoader[dict[str, BaseShard]]):
    r"""Implement a ``ShardDict`` loader."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, uri: str) -> ShardDict:
        return ShardDict.from_uri(uri)
