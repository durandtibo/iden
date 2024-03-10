r"""Contain a data structure to manage a list of shards."""

from __future__ import annotations

__all__ = ["ShardList"]

import copy
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils import repr_indent, repr_sequence, str_indent, str_sequence

if TYPE_CHECKING:
    from iden.shard.base import BaseShard


class ShardList:
    r"""Implement a data structure to manage a list of shards.

    Args:
        shards: The list of shards.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard import create_json_shard
    >>> from iden.shard.collection import ShardList
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     shards = [
    ...         create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()),
    ...         create_json_shard(
    ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
    ...         ),
    ...     ]
    ...     sl = ShardList(shards)
    ...     sl.append(
    ...         create_json_shard([8, 9], uri=Path(tmpdir).joinpath("shard/uri3").as_uri())
    ...     )
    ...     sl
    ...
    ShardList(
      (0): JsonShard(uri=file:///.../shard/uri1)
      (1): JsonShard(uri=file:///.../shard/uri2)
      (2): JsonShard(uri=file:///.../shard/uri3)
    )

    ```
    """

    def __init__(self, shards: list[BaseShard] | None = None) -> None:
        self._shards = copy.copy(shards) or []

    def __getitem__(self, index: int) -> BaseShard:
        return self._shards[index]

    def __len__(self) -> int:
        return len(self._shards)

    def __repr__(self) -> str:
        args = f"\n  {repr_indent(repr_sequence(self._shards))}\n" if self._shards else ""
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = f"\n  {str_indent(str_sequence(self._shards))}\n" if self._shards else ""
        return f"{self.__class__.__qualname__}({args})"

    def append(self, shard: BaseShard) -> None:
        r"""Append a shard to the list of shards.

        Args:
            shard: The shard to add.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardList
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = [
        ...         create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     ]
        ...     sl = ShardList(shards)
        ...     sl.append(
        ...         create_json_shard([8, 9], uri=Path(tmpdir).joinpath("shard/uri3").as_uri())
        ...     )
        ...     sl
        ...
        ShardList(
          (0): JsonShard(uri=file:///.../shard/uri1)
          (1): JsonShard(uri=file:///.../shard/uri2)
          (2): JsonShard(uri=file:///.../shard/uri3)
        )

        ```
        """
        self._shards.append(shard)

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
        >>> from iden.shard.collection import ShardList
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = [
        ...         create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     ]
        ...     ShardList(shards).equal(ShardList(shards))
        ...
        True

        ```
        """
        if not isinstance(other, ShardList):
            return False
        return objects_are_equal(self._shards, other._shards, equal_nan=equal_nan)

    def get(self, index: int) -> BaseShard:
        r"""Get a shard.

        Args:
            index: The shard index to get.

        Returns:
            The shard.

        Raises:
            IndexError: if the index is outside  the list range.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardList
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = [
        ...         create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     ]
        ...     sl = ShardList(shards)
        ...     sl.get(0)
        ...
        JsonShard(uri=file:///.../uri1)

        ```
        """
        return self[index]

    def get_shards(self) -> list[BaseShard]:
        r"""Get the list of shards.

        Returns:
            The list of shards.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardList
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = [
        ...         create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     ]
        ...     sl = ShardList(shards)
        ...     sl.get_shards()
        ...
        [JsonShard(uri=file:///.../uri1), JsonShard(uri=file:///.../uri2)]

        ```
        """
        return self._shards

    def pop(self, index: int) -> BaseShard:
        r"""Remove the shard at the given position in the list, and
        return it.

        Args:
            index: The index of the shard to remove.

        Returns:
            The popped shard.

        Raises:
            IndexError: if the list is empty or the index is outside
                the list range.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardList
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = [
        ...         create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     ]
        ...     sl = ShardList(shards)
        ...     sl.pop(0)
        ...     sl
        ...
        JsonShard(uri=file:///.../shard/uri1)
        ShardList(
          (0): JsonShard(uri=file:///.../shard/uri2)
        )

        ```
        """
        return self._shards.pop(index)

    def get_uris(self) -> list[str]:
        r"""Get the list of shard's URI.

        Returns:
            The list of shard's URI.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import create_json_shard
        >>> from iden.shard.collection import ShardList
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shards = [
        ...         create_json_shard(
        ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
        ...         ),
        ...         create_json_shard(
        ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
        ...         ),
        ...     ]
        ...     sl = ShardList(shards)
        ...     sl.get_uris()
        ...
        ['file:///.../shard/uri1', 'file:///.../shard/uri2']

        ```
        """
        return [shard.get_uri() for shard in self._shards]
