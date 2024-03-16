r"""Contain ``VanillaDataset`` generator implementations."""

from __future__ import annotations

__all__ = ["VanillaDatasetGenerator"]

from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from iden.dataset import VanillaDataset, create_vanilla_dataset
from iden.dataset.generator import BaseDatasetGenerator
from iden.shard import BaseShard
from iden.shard.generator.base import setup_shard_generator

if TYPE_CHECKING:
    from pathlib import Path

    from iden.shard.generator import ShardDictGenerator


class VanillaDatasetGenerator(BaseDatasetGenerator[tuple[BaseShard, ...]]):
    r"""Implement a ``VanillaDataset`` generator.

    Args:
        path_uri: The path where to save the URI file.
        shards: The shards generator or its configuration.
        assets: The assets generator or its configuration.
    """

    def __init__(
        self,
        path_uri: Path,
        shards: ShardDictGenerator | dict,
        assets: ShardDictGenerator | dict,
    ) -> None:
        self._path_uri = path_uri
        self._shards = setup_shard_generator(shards)
        self._assets = setup_shard_generator(assets)

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

    def generate(self, dataset_id: str) -> VanillaDataset:
        shards = self._shards.generate(shard_id="shards")
        assets = self._assets.generate(shard_id="assets")
        return create_vanilla_dataset(
            uri=self._path_uri.joinpath(dataset_id).as_uri(),
            shards=shards,
            assets=assets,
        )
