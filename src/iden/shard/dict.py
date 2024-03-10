r"""Contain a data structure to manage a dictionary of shards."""

from __future__ import annotations

__all__ = ["ShardDict"]

import logging
from typing import Any

from coola import objects_are_equal
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from objectory import OBJECT_TARGET

from iden.constants import LOADER, SHARDS
from iden.io import JsonSaver
from iden.shard.base import BaseShard
from iden.shard.exceptions import ShardNotFoundError
from iden.shard.utils import get_dict_uris
from iden.utils.path import sanitize_path

logger = logging.getLogger(__name__)


class ShardDict(BaseShard):
    r"""Implement a data structure to manage a dictionary of shards.

    Args:
        uri: The shard's URI.
        shards: The dictionary of shards.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.dataset import VanillaDataset
    >>> from iden.shard import create_json_shard, ShardDict
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     shards = {
    ...         "train": create_json_shard(
    ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
    ...         ),
    ...         "val": create_json_shard(
    ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
    ...         ),
    ...     }
    ...     sd = ShardDict(uri=Path(tmpdir).joinpath("main_uri").as_uri(), shards=shards)
    ...     print(sd)
    ...
    ShardDict(
      (train): JsonShard(uri=file:///.../shard/uri1)
      (val): JsonShard(uri=file:///.../shard/uri2)
    )

    ```
    """

    def __init__(self, uri: str, shards: dict[str, BaseShard]) -> None:
        self._uri = uri
        self._shards = shards.copy()

    def __len__(self) -> int:
        return len(self._shards)

    def __repr__(self) -> str:
        args = f"\n  {repr_indent(repr_mapping(self._shards))}\n" if self._shards else ""
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = f"\n  {str_indent(str_mapping(self._shards))}\n" if self._shards else ""
        return f"{self.__class__.__qualname__}({args})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(
            self.get_uri(), other.get_uri(), equal_nan=equal_nan
        ) and objects_are_equal(self.get_data(), other.get_data(), equal_nan=equal_nan)

    def get_data(self) -> dict[str, BaseShard]:
        return self._shards.copy()

    def get_uri(self) -> str:
        return self._uri

    def get_shard(self, shard_id: str) -> Any:
        r"""Get a shard.

        Args:
            shard_id: The shard ID.

        Returns:
            The shard.

        Raises:
            ShardNotFoundError: if the shard does not exist.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard, ShardDict
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = {
        ...         "train": create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         "val": create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     }
        ...     sd = ShardDict(uri=Path(tmpdir).joinpath("main_uri").as_uri(), shards=shards)
        ...     print(sd.get_shard("train"))
        ...
        JsonShard(uri=file:///.../uri1)

        ```
        """
        if shard_id not in self._shards:
            msg = f"shard `{shard_id}` does not exist"
            raise ShardNotFoundError(msg)
        return self._shards[shard_id]

    def has_shard(self, shard_id: str) -> bool:
        r"""Indicate if the shard exists or not.

        Args:
            shard_id: The shard ID.

        Returns:
            ``True`` if the shard exists, otherwise ``False``

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard, ShardDict
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = {
        ...         "train": create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         "val": create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     }
        ...     sd = ShardDict(uri=Path(tmpdir).joinpath("main_uri").as_uri(), shards=shards)
        ...     print(sd.has_shard("train"))
        ...     print(sd.has_shard("test"))
        ...
        True
        False

        ```
        """
        return shard_id in self._shards

    @classmethod
    def generate_uri_config(cls, shards: dict[str, BaseShard]) -> dict:
        r"""Generate the minimal config that is used to load the shard
        from its URI.

        The config must be compatible with the JSON format.

        Args:
            shards: The shards.

        Returns:
            The minimal config to load the shard from its URI.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import ShardDict, create_json_shard
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = {
        ...         "train": create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         "val": create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     }
        ...     ShardDict.generate_uri_config(shards)
        ...
        {'shards': {'train': 'file:///.../shard/uri1', 'val': 'file:///.../shard/uri2'},
         'loader': {'_target_': 'iden.shard.loader.ShardDictLoader'}}

        ```
        """
        return {
            SHARDS: get_dict_uris(shards),
            LOADER: {OBJECT_TARGET: "iden.shard.loader.ShardDictLoader"},
        }


def create_shard_dict(shards: dict[str, BaseShard], uri: str) -> ShardDict:
    r"""Create a ``ShardDict`` a list of shards.

    Note:
        It is a utility function to create a ``ShardDict`` from its
            shards and URI. It is possible to create a ``ShardDict``
            in other ways.

    Args:
        shards: The shards.
        uri: The shard's URI.

    Returns:
        The ``ShardDict`` object.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard import ShardDict, create_json_shard, create_shard_dict
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     shards = {
    ...         "train": create_json_shard(
    ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
    ...         ),
    ...         "val": create_json_shard(
    ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
    ...         ),
    ...     }
    ...     shard = create_shard_dict(shards, uri=Path(tmpdir).joinpath("my_uri").as_uri())
    ...     print(shard)
    ...
    ShardDict(
      (train): JsonShard(uri=file:///.../shard/uri1)
      (val): JsonShard(uri=file:///.../shard/uri2)
    )

    ```
    """
    logger.info(f"Saving URI file {uri}")
    JsonSaver().save(ShardDict.generate_uri_config(shards), sanitize_path(uri))
    return ShardDict(uri, shards)
