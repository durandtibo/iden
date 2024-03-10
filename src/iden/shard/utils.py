r"""Contain code to load a shard from its Uniform Resource Identifier
(URI)."""

from __future__ import annotations

__all__ = ["get_dict_uris", "get_list_uris", "sort_by_uri"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from iden.shard.base import BaseShard


def get_dict_uris(shards: dict[str, BaseShard]) -> dict[str, str]:
    r"""Get the dictionary of shard's URI.

    Args:
        shards: The dictionary of shards.

    Returns:
        The dictionary of shard's URI.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard import create_json_shard, get_dict_uris
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     shards = {
    ...         "train": create_json_shard(
    ...             [1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()
    ...         ),
    ...         "val": create_json_shard(
    ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
    ...         ),
    ...     }
    ...     get_dict_uris(shards)
    ...
    {'train': 'file:///.../shard/uri1', 'val': 'file:///.../shard/uri2'}

    ```
    """
    return {key: shard.get_uri() for key, shard in shards.items()}


def get_list_uris(shards: Iterable[BaseShard]) -> list[str]:
    r"""Get the list of shard's URI.

    Args:
        shards: The shards.

    Returns:
        The tuple of shard's URI.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard import get_list_uris, create_json_shard
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     shards = [
    ...         create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("shard/uri1").as_uri()),
    ...         create_json_shard(
    ...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shard/uri2").as_uri()
    ...         ),
    ...     ]
    ...     get_list_uris(shards)
    ...
    ['file:///.../shard/uri1', 'file:///.../shard/uri2']

    ```
    """
    return [shard.get_uri() for shard in shards]


def sort_by_uri(shards: Iterable[BaseShard], /, *, reverse: bool = False) -> list[BaseShard]:
    r"""Sort a sequence of shards by their URIs.

    Args:
        shards: The shards to sort.
        reverse: If set to ``True``, then the list elements are sorted
            as if each comparison were reversed.

    Returns:
        The sorted shards.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard import create_json_shard, sort_by_uri
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     shards = sort_by_uri(
    ...         [
    ...             create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("uri2").as_uri()),
    ...             create_json_shard([4, 5, 6, 7], uri=Path(tmpdir).joinpath("uri3").as_uri()),
    ...             create_json_shard([4, 5, 6, 7], uri=Path(tmpdir).joinpath("uri1").as_uri()),
    ...         ]
    ...     )
    ...     shards
    ...
    [JsonShard(uri=file:///.../uri1), JsonShard(uri=file:///.../uri2), JsonShard(uri=file:///.../uri3)]

    ```
    """
    return sorted(shards, key=lambda item: item.get_uri(), reverse=reverse)
