r"""Contain ``VanillaDataset`` creator implementations."""

from __future__ import annotations

__all__ = ["VanillaDatasetCreator"]

from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from iden.dataset import VanillaDataset, create_vanilla_dataset
from iden.dataset.creator import BaseDatasetCreator
from iden.shard import BaseShard
from iden.shard.creator.base import setup_shard_creator

if TYPE_CHECKING:
    from pathlib import Path

    from iden.shard.creator import ShardDictCreator


class VanillaDatasetCreator(BaseDatasetCreator[tuple[BaseShard, ...]]):
    r"""Implement a ``VanillaDataset`` creator.

    Args:
        path_uri: The path where to save the URI file.
        shards: The shards creator or its configuration.
        assets: The assets creator or its configuration.
    """

    def __init__(
        self,
        path_uri: Path,
        shards: ShardDictCreator | dict,
        assets: ShardDictCreator | dict,
    ) -> None:
        self._path_uri = path_uri
        self._shards = setup_shard_creator(shards)
        self._assets = setup_shard_creator(assets)

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "path_uri": self._path_uri,
                    "shards": self._shards,
                    "assets": self._assets,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "path_uri": self._path_uri,
                    "shards": self._shards,
                    "assets": self._assets,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def create(self, dataset_id: str) -> VanillaDataset:
        shards = self._shards.create(shard_id="shards")
        assets = self._assets.create(shard_id="assets")
        return create_vanilla_dataset(
            uri=self._path_uri.joinpath(dataset_id).as_uri(),
            shards=shards,
            assets=assets,
        )
