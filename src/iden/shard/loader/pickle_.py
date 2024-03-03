r"""Contain the pickle shard loader implementations."""

from __future__ import annotations

__all__ = ["PickleShardLoader"]

from typing import TypeVar

from iden.shard.loader.base import BaseShardLoader
from iden.shard.pickle_ import PickleShard

T = TypeVar("T")


class PickleShardLoader(BaseShardLoader[T]):
    r"""Implement a pickle shard loader."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, uri: str) -> PickleShard[T]:
        return PickleShard.from_uri(uri)
