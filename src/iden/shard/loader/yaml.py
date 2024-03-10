r"""Contain YAML shard loader implementations."""

from __future__ import annotations

__all__ = ["YamlShardLoader"]

from typing import Any, TypeVar

from iden.shard.loader.base import BaseShardLoader
from iden.shard.yaml import YamlShard

T = TypeVar("T")


class YamlShardLoader(BaseShardLoader[Any]):
    r"""Implement a YAML shard loader."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, uri: str) -> YamlShard:
        return YamlShard.from_uri(uri)
