r"""Contain the base class to implement a shard object."""

from __future__ import annotations

__all__ = ["BaseShard"]

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class BaseShard(Generic[T], ABC):
    r"""Define the base class to implement a shard."""

    @abstractmethod
    def clear(self) -> None:
        r"""Clear the current shard i.e. remove from memory the data if
        possible."""

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two shards are equal or not.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

        Returns:
            ``True`` if the two shards are equal, otherwise ``False``.
        """

    @abstractmethod
    def get_data(self) -> T:
        r"""Get the data in the shard.

        Returns:
            The data in the shard.
        """

    @abstractmethod
    def get_uri(self) -> str | None:
        r"""Get the Uniform Resource Identifier (URI) of the shard.

        Returns:
            The Uniform Resource Identifier (URI).
        """

    @abstractmethod
    def is_initialized(self) -> bool:
        r"""Indicate if the shard has data in-memory or not.

        Returns:
            ``True`` if the shard has data in-memory otherwise ``False``.
        """
