r"""Contain the base class to implement a dataset object."""

from __future__ import annotations

__all__ = ["BaseDataset"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable

    from iden.shard import BaseShard

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataset(Generic[T], ABC):
    r"""Define the base class to implement a dataset.

    Note this dataset class is very different from the PyTorch dataset
    class because it has a different goal. One of the goals is to help
    to organize and manage shards.
    """

    @abstractmethod
    def get_asset(self, asset_id: str) -> Any:
        r"""Get a data asset from this sharded dataset.

        This method is useful to access some data variables/parameters
        that are not available before to load/preprocess the data.

        Args:
            asset_id: Specifies the ID of the asset.

        Returns:
            The asset.

        Raises:
            AssetNotFoundError: if the asset does not exist.

        Example:
        ```pycon
        >>> from iden.dataset import BaseDataset
        >>> dataset: BaseDataset = ...  # Instantiate a sharded dataset
        >>> my_asset = dataset.get_asset("my_asset_id")

        ```
        """

    @abstractmethod
    def has_asset(self, asset_id: str) -> bool:
        r"""Indicate if the asset exists or not.

        Args:
            asset_id: Specifies the ID of the asset.

        Returns:
            ``True`` if the asset exists, otherwise ``False``.

        Example:
        ```pycon
        >>> from iden.dataset import BaseDataset
        >>> dataset: BaseDataset = ...  # Instantiate a sharded dataset
        >>> dataset.has_asset('my_asset_id')
        True  # If the asset exists
        >>> dataset.has_asset('my_asset_id')
        False  # If the asset does not exist

        ```
        """

    @abstractmethod
    def get_shards(self, split: str) -> Iterable[BaseShard[T]]:
        r"""Get the shards for a given split.

        Returns:
            iterable: The shards for a given split. The shards are
                sorted by ascending order of URI.

        Raises:
            ``SplitNotFoundError``: if the split does not exist.

        Example:
        ```pycon
        >>> from iden.dataset import BaseDataset
        >>> dataset: BaseDataset = ...  # Instantiate a sharded dataset
        >>> train_shards = dataset.get_shards('train')
        >>> val_shards = dataset.get_shards('val')

        ```
        """

    @abstractmethod
    def get_metadata(self, split: str) -> dict:
        r"""Get the metadata for a given split.

        The values in the metadata dict depends on the sharded dataset
        implementation.

        Returns:
            The sharded dataset metadata.

        Raises:
            ``SplitNotFoundError``: if the split does not exist.

        Example:
        ```pycon
        >>> from iden.dataset import BaseDataset
        >>> dataset: BaseDataset = ...  # Instantiate a sharded dataset
        >>> dataset.get_metadata('train')
        {...}

        ```
        """

    @abstractmethod
    def get_num_shards(self, split: str) -> int:
        r"""Get the number of shards for a given split.

        Returns:
            The number of shards in the dataset for a given split.

        Raises:
            ``SplitNotFoundError``: if the split does not exist.

        Example:
        ```pycon
        >>> from iden.dataset import BaseDataset
        >>> dataset: BaseDataset = ...  # Instantiate a sharded dataset
        >>> dataset.get_num_shards('train')
        12  # Assume there are 12 shards for the train split

        ```
        """

    @abstractmethod
    def get_splits(self) -> set[str]:
        r"""Get the available dataset splits.

        Returns:
            The dataset splits.

        Example:
        ```pycon
        >>> from iden.dataset import BaseDataset
        >>> dataset: BaseDataset = ...  # Instantiate a sharded dataset
        >>> dataset.get_splits()
        {'train', 'val'}  # Assume the splits are 'train' and 'val'

        ```
        """

    @abstractmethod
    def has_split(self, split: str) -> bool:
        r"""Indicate if a dataset split exists or not.

        Returns:
            ``True`` of the split exists, otherwise ``False``

        Example:
        ```pycon
        >>> from iden.dataset import BaseDataset
        >>> dataset: BaseDataset = ...  # Instantiate a sharded dataset
        >>> dataset.has_split('train')
        True  # Assume the train split exists
        >>> dataset.has_split('test')
        False  # Assume the tes split does not exist

        ```
        """


class SplitNotFoundError(Exception):
    r"""Raised when trying to access a split that does not exist."""
