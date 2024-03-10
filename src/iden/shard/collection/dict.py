r"""Contain a data structure to manage a dictionary of shards."""

from __future__ import annotations

__all__ = ["ShardDict"]

import copy
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from iden.shard.exceptions import ShardExistsError, ShardNotFoundError

if TYPE_CHECKING:
    from iden.shard.base import BaseShard


class ShardDict:
    r"""Implement a data structure to manage a dictionary of shards.

    Args:
        shards: The dictionary of shards.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.dataset import VanillaDataset
    >>> from iden.shard import create_json_shard
    >>> from iden.shard.collection import ShardDict
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     shards = {
    ...         "train": create_json_shard(
    ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
    ...         ),
    ...         "val": create_json_shard(
    ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
    ...         ),
    ...     }
    ...     sd = ShardDict(shards)
    ...     shard = create_json_shard([8, 9], uri=Path(tmpdir).joinpath("shard/uri3").as_uri())
    ...     sd.add_shard("test", shard)
    ...     sd
    ...
    ShardDict(
      (train): JsonShard(uri=file:///.../shard/uri1)
      (val): JsonShard(uri=file:///.../shard/uri2)
      (test): JsonShard(uri=file:///.../shard/uri3)
    )

    ```
    """

    def __init__(self, shards: dict[str, BaseShard] | None = None) -> None:
        self._shards = copy.copy(shards) or {}

    def __len__(self) -> int:
        return len(self._shards)

    def __repr__(self) -> str:
        args = f"\n  {repr_indent(repr_mapping(self._shards))}\n" if self._shards else ""
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = f"\n  {str_indent(str_mapping(self._shards))}\n" if self._shards else ""
        return f"{self.__class__.__qualname__}({args})"

    def add_shard(self, shard_id: str, shard: BaseShard, replace_ok: bool = False) -> None:
        r"""Add a shard to the shard collection.

        Note that the name should be unique. If the name exists, the
        old shard will be overwritten by the new shard.

        Args:
            shard_id: The shard ID.
            shard: The shard to add to the collection.
            replace_ok: If ``False``, ``ShardExistsError`` is raised
                if a shard with the same ID exists.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardDict
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = {
        ...         "train": create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         "val": create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     }
        ...     sd = ShardDict(shards)
        ...     sd.add_shard("test",
        ...         create_json_shard([8, 9], uri=Path(tmpdir).joinpath("shard/uri3").as_uri())
        ...     )
        ...     sd.has_shard("test")
        ...
        True

        ```
        """
        if shard_id in self._shards and not replace_ok:
            msg = (
                f"`{shard_id}` is already used to register a shard. "
                "Use `replace_ok=True` to replace a shard"
            )
            raise ShardExistsError(msg)
        self._shards[shard_id] = shard

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two objects are equal.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example usage:

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardDict
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = {
        ...         "train": create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         "val": create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     }
        ...     ShardDict(shards).equal(ShardDict(shards))
        ...
        True

        ```
        """
        if not isinstance(other, ShardDict):
            return False
        return objects_are_equal(self._shards, other._shards, equal_nan=equal_nan)

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
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardDict
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = {
        ...         "train": create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         "val": create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     }
        ...     sd = ShardDict(shards)
        ...     sd.get_shard("train")
        ...
        JsonShard(uri=file:///.../uri1)

        ```
        """
        if shard_id not in self._shards:
            msg = f"shard `{shard_id}` does not exist"
            raise ShardNotFoundError(msg)
        return self._shards[shard_id]

    def get_shards(self) -> dict[str, BaseShard]:
        r"""Get the dictionary of shards.

        Returns:
            The dictionary of shards.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardDict
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = {
        ...         "train": create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         "val": create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     }
        ...     sd = ShardDict(shards)
        ...     sd.get_shards()
        ...
        {'train': JsonShard(uri=file:///.../uri1), 'val': JsonShard(uri=file:///.../uri2)}

        ```
        """
        return self._shards

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
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardDict
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = {
        ...         "train": create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         "val": create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     }
        ...     sd = ShardDict(shards)
        ...     sd.has_shard("train")
        ...     sd.has_shard("test")
        ...
        True
        False

        ```
        """
        return shard_id in self._shards

    def remove_shard(self, shard_id: str) -> None:
        r"""Remove a shard.

        Args:
            shard_id: The shard ID.

        Raises:
            ShardNotFoundError: if the shard does not exist.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardDict
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = {
        ...         "train": create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         "val": create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     }
        ...     sd = ShardDict(shards)
        ...     sd.remove_shard("train")
        ...     sd.has_shard("train")
        ...
        False

        ```
        """
        if shard_id not in self._shards:
            msg = f"shard `{shard_id}` does not exist so it is not possible to remove it"
            raise ShardNotFoundError(msg)
        del self._shards[shard_id]

    def get_uris(self) -> dict[str, str]:
        r"""Get the dictionary of shard's URI.

        Returns:
            The dictionary of shard's URI.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardDict
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = {
        ...         "train": create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         "val": create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     }
        ...     sd = ShardDict(shards)
        ...     sd.get_uris()
        ...
        {'train': 'file:///.../shard/uri1', 'val': 'file:///.../shard/uri2'}

        ```
        """
        return {key: shard.get_uri() for key, shard in self._shards.items()}
