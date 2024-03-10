r"""Contain shard loader implementations for
``VanillaDatasetLoader``."""

from __future__ import annotations

__all__ = ["VanillaDatasetLoader"]

from typing import TypeVar

from iden.dataset.loader.base import BaseDatasetLoader
from iden.dataset.vanilla import VanillaDataset

T = TypeVar("T")


class VanillaDatasetLoader(BaseDatasetLoader[T]):
    r"""Implement a ``VanillaDatasetLoader`` loader."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, uri: str) -> VanillaDataset[T]:
        return VanillaDataset.from_uri(uri)
