r"""Contain the base class to implement a dataset object."""

from __future__ import annotations

__all__ = ["VanillaDataset", "prepare_shards"]

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from coola import objects_are_equal
from coola.utils import str_indent, str_mapping
from objectory import OBJECT_TARGET

from iden.constants import LOADER, SHARDS
from iden.dataset.base import BaseDataset
from iden.dataset.exceptions import AssetNotFoundError, SplitNotFoundError
from iden.io import JsonSaver
from iden.shard import sort_by_uri
from iden.utils.path import sanitize_path

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from iden.shard import BaseShard

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VanillaDataset(BaseDataset[T]):
    r"""Implement a simple dataset.

    Args:
        uri: The URI associated to the dataset.
        shards: The dataset's shards. Each item in the mapping
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
    ...             create_json_shard(
    ...                 [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
    ...             ),
    ...             create_json_shard(
    ...                 [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
    ...             ),
    ...         ],
    ...         "val": [],
    ...     }
    ...     dataset = VanillaDataset(uri=Path(tmpdir).joinpath("uri").as_uri(), shards=shards)
    ...     dataset
    ...
    VanillaDataset(
      (uri): file:///.../uri
      (shards):
        (train): 2
        (val): 0
      (assets): []
    )

    ```
    """

    def __init__(
        self, uri: str, shards: Mapping[str, Iterable[BaseShard[T]]], assets: dict[str, Any] = None
    ) -> None:
        self._uri = str(uri)
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
                    "uri": self._uri,
                    "shards": shards,
                    "assets": sorted(self._assets.keys()),
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            objects_are_equal(self.get_uri(), other.get_uri(), equal_nan=equal_nan)
            and objects_are_equal(self._shards, other._shards, equal_nan=equal_nan)
            and objects_are_equal(self._assets, other._assets, equal_nan=equal_nan)
        )

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

    def get_uri(self) -> str:
        return self._uri

    @classmethod
    def generate_uri_config(cls, shards: Mapping[str, Iterable[BaseShard[T]]]) -> dict:
        r"""Generate the minimal config that is used to load the dataset
        from its URI.

        The config must be compatible with the JSON format.

        Args:
            shards: The shards in the dataset. Each item in the mapping
                represent a dataset split, where the key is the dataset
                split and the value is the shards.

        Returns:
            The minimal config to load the shard from its URI.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.dataset import VanillaDataset
        >>> from iden.shard import create_json_shard
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = {
        ...         "train": [
        ...             create_json_shard(
        ...                 [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...             ),
        ...             create_json_shard(
        ...                 [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...             ),
        ...         ],
        ...         "val": [],
        ...     }
        ...     config = VanillaDataset.generate_uri_config(shards)
        ...     config
        ...
        {'loader': {'_target_': 'iden.dataset.loader.VanillaDatasetLoader'},
         'shards': {'train': ('file:///.../uri1', 'file:///.../uri2'), 'val': ()}}

        ```
        """
        return {
            LOADER: {OBJECT_TARGET: "iden.dataset.loader.VanillaDatasetLoader"},
            SHARDS: {
                split: tuple(shard.get_uri() for shard in data) for split, data in shards.items()
            },
        }


def create_vanilla_dataset(
    shards: Mapping[str, Iterable[BaseShard[T]]],
    uri: str,
    assets: dict[str, Any] = None,
) -> VanillaDataset:
    r"""Create a ``VanillaDataset`` from its shards.

    Note:
        It is a utility function to create a ``VanillaDataset`` from
            its shards and URI. It is possible to create a
            ``VanillaDataset`` in other ways.

    Args:
        shards: The dataset's shards. Each item in the mapping
            represent a dataset split, where the key is the dataset
            split and the value is the shards.
        uri: The URI associated to the dataset.
        assets: The dataset's assets.

    Returns:
        The ``JsonShard`` object.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard import create_json_shard
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     shard = create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("my_uri").as_uri())
    ...     shard.get_data()
    ...
    [1, 2, 3]

    ```
    """
    logger.info(f"Saving URI file {uri}")
    JsonSaver().save(VanillaDataset.generate_uri_config(shards), sanitize_path(uri))
    return VanillaDataset(uri=uri, shards=shards, assets=assets)


def prepare_shards(
    shards: Mapping[str, Iterable[BaseShard[T]]]
) -> dict[str, tuple[BaseShard[T], ...]]:
    r"""Prepare the shards.

    The shards are sorted by ascending order of URIs.

    Args:
        shards: The shards to prepare. Each item in the mapping
            represent a dataset split, where the key is the dataset
            split and the value is the shards.

    Returns:
        The prepared shards.
    """
    return {split: tuple(sort_by_uri(data)) for split, data in shards.items()}
