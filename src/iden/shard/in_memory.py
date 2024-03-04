r"""Contain in-memory shard implementations."""

from __future__ import annotations

__all__ = ["InMemoryShard"]

from typing import Any, TypeVar

from coola import objects_are_equal

from iden.shard.base import BaseShard

T = TypeVar("T")


class InMemoryShard(BaseShard[T]):
    r"""Implement an in-memory shard.

    This shard does not have valid URI as the data are stored
    in-memory.

    Example usage:

    ```pycon
    >>> from iden.shard import InMemoryShard
    >>> shard = InMemoryShard([1, 2, 3])
    >>> shard.get_data()
    [1, 2, 3]

    ```
    """

    def __init__(self, data: T) -> None:
        self._data = data

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self._data, other._data)

    def get_data(self) -> T:
        return self._data

    def get_uri(self) -> str | None:
        return None
