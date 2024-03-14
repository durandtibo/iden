r"""Contain file-based shard creator implementations."""

from __future__ import annotations

__all__ = ["BaseFileShardCreator"]

from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from iden.shard.creator.base import BaseShardCreator

if TYPE_CHECKING:
    from pathlib import Path

    from iden.shard import JsonShard

T = TypeVar("T")


class BaseFileShardCreator(BaseShardCreator[T]):
    r"""Implement a JSON shard creator.

    Args:
        data: The data to save in the shard.
        path_uri: The path where to save the URI file.
        path_shard: The path where to save the shard data.
    """

    def __init__(self, data: T, path_uri: Path, path_shard: Path) -> None:
        self._data = data
        self._path_uri = path_uri
        self._path_shard = path_shard

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping({"path_uri": self._path_uri, "path_shard": self._path_shard})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"path_uri": self._path_uri, "path_shard": self._path_shard}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def create(self, shard_id: str) -> JsonShard[T]:
        data = self._data
        return self._create(data=data, shard_id=shard_id)

    @abstractmethod
    def _create(self, data: T, shard_id: str) -> JsonShard[T]:
        r"""Create a shard based on the data and shard ID.

        Args:
            data: The data to save in the shard.
            shard_id: The shard IDI.

        Returns:
            The created shard.
        """
