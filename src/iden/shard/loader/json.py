r"""Contain JSON shard loader implementations."""

from __future__ import annotations

__all__ = ["JsonShardLoader"]

from typing import Any, TypeVar

from iden.shard.json import JsonShard
from iden.shard.loader.base import BaseShardLoader

T = TypeVar("T")


class JsonShardLoader(BaseShardLoader[Any]):
    r"""Implement a JSON shard loader."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, uri: str) -> JsonShard:
        return JsonShard.from_uri(uri)
