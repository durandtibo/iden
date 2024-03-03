r"""Contain the base class to implement a dataset object."""

from __future__ import annotations

__all__ = ["VanillaDataset"]

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from iden.dataset.base import BaseDataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from iden.shard import BaseShard

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VanillaDataset(BaseDataset[T]):

    def __init__(self, shards: dict[str, Sequence[BaseShard[T]]], assets: dict[str, Any]) -> None:
        self._shards = shards
        self._assets = assets

    def get_asset(self, asset_id: str) -> Any:
        return self._assets[asset_id]

    def has_asset(self, asset_id: str) -> bool:
        return asset_id in self._assets

    def get_shards(self, split: str) -> Iterable[BaseShard[T]]:
        return self._shards[split]

    def get_metadata(self, split: str) -> dict:
        pass

    def get_num_shards(self, split: str) -> int:
        return len(self._shards[split])

    def get_splits(self) -> set[str]:
        return set(self._shards.keys())

    def has_split(self, split: str) -> bool:
        return split in self._shards
