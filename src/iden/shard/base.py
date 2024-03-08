r"""Contain the base class to implement a shard object."""

from __future__ import annotations

__all__ = ["BaseShard"]

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class BaseShard(Generic[T], ABC):
    r"""Define the base class to implement a shard."""

    @abstractmethod
    def equal(self, other: Any) -> bool:
        r"""Indicate if two shards are equal or not.

        Args:
            other: The object to compare with.

        Returns:
            ``True`` if the two shards are equal, otherwise ``False``.
        """

    @abstractmethod
    def get_data(self) -> T:
        r"""Get the data in the shard.

        Returns:
            The data in the shard.
        """
        # TODO(tibo): maybe we can add an option to copy data  # noqa: TD003

    @abstractmethod
    def get_uri(self) -> str | None:
        r"""Get the Uniform Resource Identifier (URI) of the shard.

        Returns:
            The Uniform Resource Identifier (URI).
        """
