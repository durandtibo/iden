r"""Contain dict-based shard generator implementations."""

from __future__ import annotations

__all__ = ["ShardDictGenerator"]

from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from iden.shard import BaseShard, ShardDict, create_shard_dict
from iden.shard.generator.base import BaseShardGenerator, setup_shard_generator

if TYPE_CHECKING:
    from pathlib import Path


class ShardDictGenerator(BaseShardGenerator[tuple[BaseShard, ...]]):
    r"""Implement a ``ShardDict`` generator.

    Args:
        shards: The shard generators or their configurations.
        path_uri: The path where to save the URI file.
    """

    def __init__(self, shards: dict[str, BaseShardGenerator | dict], path_uri: Path) -> None:
        self._shards = {key: setup_shard_generator(shard) for key, shard in shards.items()}
        self._path_uri = path_uri

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "path_uri": self._path_uri,
                    "shards": "\n" + repr_mapping(self._shards),
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "path_uri": self._path_uri,
                    "shards": "\n" + str_mapping(self._shards),
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, shard_id: str) -> ShardDict:
        shards = {key: shard.generate(str(key)) for key, shard in self._shards.items()}
        return create_shard_dict(uri=self._path_uri.joinpath(shard_id).as_uri(), shards=shards)
