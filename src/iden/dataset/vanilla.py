r"""Contain the base class to implement a dataset object."""

from __future__ import annotations

__all__ = ["VanillaDataset", "prepare_shards"]

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from coola.utils import str_indent, str_mapping

from iden.dataset.base import BaseDataset
from iden.dataset.exceptions import AssetNotFoundError, SplitNotFoundError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from iden.shard import BaseShard

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VanillaDataset(BaseDataset[T]):
    r"""Implement a simple dataset.

    Args:
        shards: The shards to prepare. Each item in the mapping
            represent a dataset split, where the key is the dataset
            split and the value is the shards.
        assets: The dataset's assets.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.dataset import VanillaDataset
    >>> from iden.shard import create_json_shard
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     shards = {
    ...         "train": [
    ...             create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("uri1").as_uri()),
    ...             create_json_shard([4, 5, 6, 7], uri=Path(tmpdir).joinpath("uri2").as_uri()),
    ...         ],
    ...         "val": [],
    ...     }
    ...     dataset = VanillaDataset(shards)
    ...     print(dataset)
    ...
    VanillaDataset(
      (shards):
        (train): 2
        (val): 0
      (assets): []
    )

    ```
    """

    def __init__(
        self, shards: Mapping[str, Iterable[BaseShard[T]]], assets: dict[str, Any] = None
    ) -> None:
        self._shards = prepare_shards(shards)
        self._assets = assets or {}

    def __repr__(self) -> str:
        shards = str_indent(
            str_mapping({split: f"{len(shards):,}" for split, shards in self._shards.items()})
        )
        if shards:
            shards = "\n" + shards
        args = str_indent(
            str_mapping(
                {
                    "shards": shards,
                    "assets": sorted(self._assets.keys()),
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def get_asset(self, asset_id: str) -> Any:
        if asset_id not in self._assets:
            msg = f"asset '{asset_id}' does not exist"
            raise AssetNotFoundError(msg)
        return self._assets[asset_id]

    def has_asset(self, asset_id: str) -> bool:
        return asset_id in self._assets

    def get_shards(self, split: str) -> tuple[BaseShard[T], ...]:
        if split not in self._shards:
            msg = f"split '{split}' does not exist"
            raise SplitNotFoundError(msg)
        return self._shards[split]

    def get_num_shards(self, split: str) -> int:
        if split not in self._shards:
            msg = f"split '{split}' does not exist"
            raise SplitNotFoundError(msg)
        return len(self._shards[split])

    def get_splits(self) -> set[str]:
        return set(self._shards.keys())

    def has_split(self, split: str) -> bool:
        return split in self._shards


def prepare_shards(
    shards: Mapping[str, Iterable[BaseShard[T]]]
) -> dict[str, tuple[BaseShard[T], ...]]:
    r"""Prepare the shards.

    Args:
        shards: The shards to prepare. Each item in the mapping
            represent a dataset split, where the key is the dataset
            split and the value is the shards.

    Returns:
        The prepared shards.
    """
    return {
        split: tuple(sorted(data, key=lambda item: item.get_uri()))
        for split, data in shards.items()
    }
