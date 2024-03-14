r"""Contain the base class to implement a shard creator."""

from __future__ import annotations

__all__ = ["BaseShardCreator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from iden.shard import BaseShard

T = TypeVar("T")


class BaseShardCreator(Generic[T], ABC):
    r"""Define the base class to create a shard."""

    @abstractmethod
    def create(self, shard_id: str) -> BaseShard[T]:
        r"""Create a shard.

        Args:
            shard_id: The shard IDI.

        Returns:
            The created shard.
        """
