r"""Contain code to load a shard from its Uniform Resource Identifier
(URI)."""

from __future__ import annotations

__all__ = ["sort_by_uri"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from iden.shard.base import BaseShard


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
    ...     print(shards)
    ...
    [JsonShard(uri=file:///.../uri1), JsonShard(uri=file:///.../uri2), JsonShard(uri=file:///.../uri3)]

    ```
    """
    return sorted(shards, key=lambda item: item.get_uri(), reverse=reverse)
