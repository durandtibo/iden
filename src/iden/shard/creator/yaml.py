r"""Contain YAML shard creator implementations."""

from __future__ import annotations

__all__ = ["YamlShardCreator"]

from typing import TYPE_CHECKING, TypeVar

from iden.shard import YamlShard, create_yaml_shard
from iden.shard.creator.file import BaseFileShardCreator
from iden.utils.imports import check_yaml

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


class YamlShardCreator(BaseFileShardCreator[T]):
    r"""Implement a YAML shard creator.

    Args:
        data: The data to save in the shard.
        path_uri: The path where to save the URI file.
        path_shard: The path where to save the shard data.
    """

    def __init__(self, data: T, path_uri: Path, path_shard: Path) -> None:
        check_yaml()
        super().__init__(data=data, path_uri=path_uri, path_shard=path_shard)

    def _create(self, data: T, shard_id: str) -> YamlShard[T]:
        return create_yaml_shard(
            data=data,
            uri=self._path_uri.joinpath(shard_id).as_uri(),
            path=self._path_shard.joinpath(shard_id).with_suffix(".yaml"),
        )
