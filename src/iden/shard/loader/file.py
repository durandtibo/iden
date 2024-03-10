r"""Contain file-based shard loader implementations."""

from __future__ import annotations

__all__ = ["FileShardLoader"]

from typing import Any, TypeVar

from iden.shard.file import FileShard
from iden.shard.loader.base import BaseShardLoader

T = TypeVar("T")


class FileShardLoader(BaseShardLoader[Any]):
    r"""Implement a file-based shard loader."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, uri: str) -> FileShard:
        return FileShard.from_uri(uri)
