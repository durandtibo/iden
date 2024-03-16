r"""Contain the base class to implement a dataset creator."""

from __future__ import annotations

__all__ = ["BaseDatasetCreator", "is_dataset_creator_config", "setup_dataset_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory
from objectory.utils import is_object_config

if TYPE_CHECKING:
    from iden.dataset import BaseDataset

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseDatasetCreator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to create a dataset."""

    @abstractmethod
    def create(self, dataset_id: str) -> BaseDataset[T]:
        r"""Create a dataset.

        Args:
            dataset_id: The dataset IDI.

        Returns:
            The created dataset.
        """


def is_dataset_creator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseDatasetCreator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: Specifies the configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseDatasetCreator`` object.

    Example usage:

    ```pycon
    >>> from iden.dataset.creator import is_dataset_creator_config
    >>> is_dataset_creator_config({"_target_": "iden.dataset.creator.VanillaDatasetCreator"})
    True

    ```
    """
    return is_object_config(config, BaseDatasetCreator)


def setup_dataset_creator(dataset_creator: BaseDatasetCreator | dict) -> BaseDatasetCreator:
    r"""Set up a dataset creator.

    The dataset creator is instantiated from its configuration by using the
    ``BaseDatasetCreator`` factory function.

    Args:
        dataset_creator: Specifies the dataset creator or its configuration.

    Returns:
        The instantiated dataset creator.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.dataset.creator import setup_dataset_creator
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     creator = setup_dataset_creator(
    ...         {
    ...             "_target_": "iden.dataset.creator.VanillaDatasetCreator",
    ...             "path_uri": Path(tmpdir).joinpath("uri"),
    ...             "shards": {
    ...                 "_target_": "iden.shard.creator.ShardDictCreator",
    ...                 "path_uri": Path(tmpdir).joinpath("uri/shards"),
    ...                 "shards": {}
    ...             },
    ...             "assets": {
    ...                 "_target_": "iden.shard.creator.ShardDictCreator",
    ...                 "path_uri": Path(tmpdir).joinpath("uri/assets"),
    ...                 "shards": {}
    ...             }
    ...         }
    ...     )
    ...     creator
    ...
    VanillaDatasetCreator(
      (path_uri): PosixPath('/.../uri')
      (shards): ShardDictCreator(
          (path_uri): PosixPath('/.../uri/shards')
          (shards):
        )
      (assets): ShardDictCreator(
          (path_uri): PosixPath('/.../uri/assets')
          (shards):
        )
    )

    ```
    """
    if isinstance(dataset_creator, dict):
        logger.debug("Initializing a dataset creator from its configuration...")
        dataset_creator = BaseDatasetCreator.factory(**dataset_creator)
    if not isinstance(dataset_creator, BaseDatasetCreator):
        logger.warning(
            f"dataset creator is not a BaseDatasetCreator (received: {type(dataset_creator)})"
        )
    return dataset_creator
