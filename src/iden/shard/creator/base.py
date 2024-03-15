r"""Contain the base class to implement a shard creator."""

from __future__ import annotations

__all__ = ["BaseShardCreator", "is_shard_creator_config", "setup_shard_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory
from objectory.utils import is_object_config

if TYPE_CHECKING:
    from iden.shard import BaseShard

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseShardCreator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to create a shard."""

    @abstractmethod
    def create(self, shard_id: str) -> BaseShard[T]:
        r"""Create a shard.

        Args:
            shard_id: The shard IDI.

        Returns:
            The created shard.
        """


def is_shard_creator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseShardCreator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: Specifies the configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseShardCreator`` object.

    Example usage:

    ```pycon
    >>> from iden.shard.creator import is_shard_creator_config
    >>> is_shard_creator_config({"_target_": "iden.shard.creator.JsonShardCreator"})
    True

    ```
    """
    return is_object_config(config, BaseShardCreator)


def setup_shard_creator(shard_creator: BaseShardCreator | dict) -> BaseShardCreator:
    r"""Set up a shard creator.

    The shard creator is instantiated from its configuration by using the
    ``BaseShardCreator`` factory function.

    Args:
        shard_creator: Specifies the shard creator or its configuration.

    Returns:
        The instantiated shard creator.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard.creator import setup_shard_creator
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     creator = setup_shard_creator(
    ...         {
    ...             "_target_": "iden.shard.creator.JsonShardCreator",
    ...             "data": [1, 2, 3],
    ...             "path_uri": Path(tmpdir).joinpath("uri"),
    ...             "path_shard": Path(tmpdir).joinpath("data"),
    ...         }
    ...     )
    ...     creator
    ...
    JsonShardCreator(
      (path_uri): PosixPath('/.../uri')
      (path_shard): PosixPath('/.../data')
    )

    ```
    """
    if isinstance(shard_creator, dict):
        logger.debug("Initializing a shard creator from its configuration...")
        shard_creator = BaseShardCreator.factory(**shard_creator)
    if not isinstance(shard_creator, BaseShardCreator):
        logger.warning(f"shard creator is not a BaseShardCreator (received: {type(shard_creator)})")
    return shard_creator
