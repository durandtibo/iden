r"""Contain the base class to implement a shard loader object."""

from __future__ import annotations

__all__ = ["BaseShardLoader"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from iden.shard import BaseShard

T = TypeVar("T")


class BaseShardLoader(Generic[T], ABC):
    r"""Define the base class to implement a shard loader.

    A shard loader object allows to load a ``BaseShard`` object from
    its Uniform Resource Identifier (URI).
    """

    @abstractmethod
    def load(self, uri: str) -> BaseShard[T]:
        r"""Load a shard from its Uniform Resource Identifier (URI).

        Args:
            uri: The URI of the shard to load.

        Returns:
            The loaded shard.
        """
