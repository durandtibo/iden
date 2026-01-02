# Dataset

## How to implement a custom dataset?

This section explains how to extend `iden` to support custom dataset implementations.
`iden` provides a `BaseDataset` abstract class that can be extended to create custom dataset implementations.

A new dataset can be implemented by extending the `iden.dataset.BaseDataset` class and implementing the required abstract methods:

- `equal` - Compare two datasets for equality
- `get_asset` - Get an asset by name
- `get_num_shards` - Get the number of shards in a split
- `get_shards` - Get all shards for a split
- `get_splits` - Get all split names as a set
- `get_uri` - Get the dataset URI
- `has_asset` - Check if an asset exists
- `has_split` - Check if a split exists

### Example: Custom dataset implementation

Here's a minimal example of a custom dataset:

```python
from __future__ import annotations

from typing import Any

from iden.dataset.base import BaseDataset
from iden.shard import BaseShard


class CustomDataset(BaseDataset[Any]):
    """Custom dataset implementation."""

    def __init__(self, uri: str, data: dict[str, list[BaseShard]]) -> None:
        self._uri = uri
        self._data = data

    def equal(self, other: object) -> bool:
        if not isinstance(other, CustomDataset):
            return False
        return self._uri == other._uri

    def get_asset(self, asset: str) -> BaseShard:
        # Implement asset retrieval
        raise NotImplementedError

    def get_num_shards(self, split: str) -> int:
        return len(self._data.get(split, []))

    def get_shards(self, split: str) -> tuple[BaseShard[T], ...]:
        # Return shards as a tuple
        return tuple(self._data.get(split, []))

    def get_splits(self) -> set[str]:
        return set(self._data.keys())

    def get_uri(self) -> str:
        return self._uri

    def has_asset(self, asset: str) -> bool:
        return False

    def has_split(self, split: str) -> bool:
        return split in self._data
```

## How to create a dataset loader?

Dataset loaders enable instantiating datasets from their URI configuration files.
To create a custom loader, extend `iden.dataset.loader.BaseDatasetLoader`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from iden.dataset.loader.base import BaseDatasetLoader

if TYPE_CHECKING:
    from iden.dataset import BaseDataset


class CustomDatasetLoader(BaseDatasetLoader):
    """Custom dataset loader."""

    def load(self, uri: str) -> BaseDataset:
        # Load dataset configuration from URI
        # Instantiate and return the dataset
        pass
```
