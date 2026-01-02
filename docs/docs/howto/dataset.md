# Dataset

## How to implement a custom dataset?

This section explains how to extend `iden` to support custom dataset implementations.
`iden` provides a `BaseDataset` abstract class that can be extended to create custom dataset implementations.

A new dataset can be implemented by extending the `iden.dataset.BaseDataset` class and implementing the required abstract methods:

- `equal` - Compare two datasets for equality
- `get_asset` - Get an asset by name
- `get_asset_names` - Get all asset names
- `get_num_shards` - Get the number of shards in a split
- `get_shard` - Get a shard by split and index
- `get_shards` - Get all shards for a split
- `get_split_names` - Get all split names
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

    def get_asset_names(self) -> tuple[str, ...]:
        # Implement asset name retrieval
        return ()

    def get_num_shards(self, split: str) -> int:
        return len(self._data.get(split, []))

    def get_shard(self, split: str, index: int) -> BaseShard:
        return self._data[split][index]

    def get_shards(self, split: str) -> BaseShard:
        # Return a shard container (e.g., ShardTuple)
        from iden.shard import ShardTuple
        return ShardTuple(
            uri=f"{self._uri}/shards/{split}",
            shards=self._data[split],
        )

    def get_split_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._data.keys()))

    def get_uri(self) -> str:
        return self._uri

    def has_asset(self, asset: str) -> bool:
        return False

    def has_split(self, split: str) -> bool:
        return split in self._data
```

### Best practices

1. **Lazy loading**: Load data only when needed (e.g., in `get_data()` method of shards)
2. **URI management**: Use unique URIs for datasets and their components
3. **Validation**: Validate split names and asset names before accessing
4. **Error handling**: Provide clear error messages when splits or assets are not found
5. **Documentation**: Document the expected structure and behavior of your custom dataset

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

Then register your loader with the dataset loader registry:

```python
from iden.dataset.loader import get_default_dataset_loader_registry

registry = get_default_dataset_loader_registry()
registry.register_loader("custom", CustomDatasetLoader())
```
