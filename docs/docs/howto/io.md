# I/O Operations

## How to save and load data?

`iden` provides a flexible I/O system for saving and loading data in various formats.

### Using format-specific functions

The simplest way to save and load data is using format-specific convenience functions:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.io import save_json, load_json
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("data.json")
...     # Save data
...     save_json({"key": "value", "numbers": [1, 2, 3]}, path)
...     # Load data
...     data = load_json(path)
...     data
...
{'key': 'value', 'numbers': [1, 2, 3]}

```

### Supported formats

`iden` supports multiple data formats:

| Format | Save function | Load function | Required package |
|--------|---------------|---------------|------------------|
| JSON | `save_json` | `load_json` | `json` (built-in) |
| Pickle | `save_pickle` | `load_pickle` | `pickle` (built-in) |
| YAML | `save_yaml` | `load_yaml` | `pyyaml` |
| PyTorch | `save_torch` | `load_torch` | `torch` |
| Cloudpickle | `save_cloudpickle` | `load_cloudpickle` | `cloudpickle` |
| Joblib | `save_joblib` | `load_joblib` | `joblib` |
| Text | `save_text` | `load_text` | - |

### Example: YAML format

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.io import save_yaml, load_yaml
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("config.yaml")
...     config = {"model": "resnet50", "epochs": 100, "lr": 0.001}
...     save_yaml(config, path)
...     loaded = load_yaml(path)
...     loaded
...
{'model': 'resnet50', 'epochs': 100, 'lr': 0.001}

```

### Example: PyTorch tensors

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> import torch
>>> from iden.io import save_torch, load_torch
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("model.pt")
...     data = {"weights": torch.randn(10, 10), "bias": torch.zeros(10)}
...     save_torch(data, path)
...     loaded = load_torch(path)
...     loaded.keys()
...
dict_keys(['weights', 'bias'])

```

## How to use loaders and savers?

For more control, you can use loader and saver classes directly:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.io import JsonSaver, JsonLoader
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("data.json")
...     # Create saver and save data
...     saver = JsonSaver()
...     saver.save({"key": "value"}, path)
...     # Create loader and load data
...     loader = JsonLoader()
...     data = loader.load(path)
...     data
...
{'key': 'value'}

```

## How to register custom loaders?

You can register custom loaders for specific file extensions:

```pycon
>>> from iden.io import get_default_loader_registry, JsonLoader
>>> registry = get_default_loader_registry()
>>> # Register loader for .jsonl extension
>>> registry.register_loader(".jsonl", JsonLoader())
>>> # Check registered loaders
>>> registry.has_loader(".jsonl")
True

```

## How to implement a custom loader?

To create a custom loader, extend the `BaseLoader` class:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from iden.io.base import BaseLoader


class CustomLoader(BaseLoader):
    """Custom data loader."""

    def load(self, path: Path) -> Any:
        """Load data from path.

        Args:
            path: The path to load data from.

        Returns:
            The loaded data.
        """
        # Implement custom loading logic
        with open(path) as f:
            data = f.read()
        return self._process_data(data)

    def _process_data(self, data: str) -> Any:
        """Process raw data."""
        # Custom processing logic
        return data
```

## How to implement a custom saver?

To create a custom saver, extend the `BaseSaver` class:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from iden.io.base import BaseFileSaver


class CustomSaver(BaseFileSaver):
    """Custom data saver."""

    def save(self, data: Any, path: Path) -> None:
        """Save data to path.

        Args:
            data: The data to save.
            path: The path to save data to.
        """
        # Ensure parent directory exists
        self._create_parent_dir(path)

        # Implement custom saving logic
        processed = self._process_data(data)
        with open(path, "w") as f:
            f.write(processed)

    def _process_data(self, data: Any) -> str:
        """Process data before saving."""
        # Custom processing logic
        return str(data)
```

## Working with safetensors

For efficient storage of NumPy arrays or PyTorch tensors, use safetensors format:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> import torch
>>> from iden.io.safetensors import save_safetensors, load_safetensors
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("tensors.safetensors")
...     # Save tensors
...     tensors = {"layer1": torch.randn(100, 100), "layer2": torch.randn(50, 50)}
...     save_safetensors(tensors, path)
...     # Load tensors
...     loaded = load_safetensors(path)
...     loaded.keys()
...
dict_keys(['layer1', 'layer2'])

```

## Best practices

1. **Use appropriate formats**: Choose the format based on your data type and requirements
2. **Handle errors**: Always wrap I/O operations in try-except blocks
3. **Validate paths**: Ensure directories exist before saving
4. **Close resources**: Use context managers when working with files
5. **Consider compression**: For large datasets, consider compressed formats
6. **Version your data**: Include version information in saved files for compatibility
