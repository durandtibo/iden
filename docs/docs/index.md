# Home

<p align="center">
    <a href="https://github.com/durandtibo/iden/actions">
        <img alt="CI" src="https://github.com/durandtibo/iden/workflows/CI/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/iden/actions">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/iden/workflows/Nightly%20Tests/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/iden/actions">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/iden/workflows/Nightly%20Package%20Tests/badge.svg">
    </a>
    <a href="https://codecov.io/gh/durandtibo/iden">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/iden/branch/main/graph/badge.svg">
    </a>
    <br/>
    <a href="https://durandtibo.github.io/iden/">
        <img alt="Documentation" src="https://github.com/durandtibo/iden/workflows/Documentation%20(stable)/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/iden/">
        <img alt="Documentation" src="https://github.com/durandtibo/iden/workflows/Documentation%20(unstable)/badge.svg">
    </a>
    <br/>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;">
    </a>
    <a href="https://github.com/guilatrova/tryceratops">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/try%2Fexcept%20style-tryceratops%20%F0%9F%A6%96%E2%9C%A8-black">
    </a>
    <br/>
    <a href="https://pypi.org/project/iden/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/iden">
    </a>
    <a href="https://pypi.org/project/iden/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/iden.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/iden">
    </a>
    <br/>
    <a href="https://pepy.tech/project/iden">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/iden">
    </a>
    <a href="https://pepy.tech/project/iden">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/iden/month">
    </a>
    <br/>
</p>

## Overview

`iden` is a simple Python library to manage a dataset of shards when training a machine learning
model.
`iden` uses a lazy loading approach to load the shard's data, so it is easy to manage shards without
loading their data.
`iden` supports different formats to store shards on disk.

### Key Features

- **Lazy Loading**: Shards are loaded only when needed, enabling efficient memory management
- **Multiple Formats**: Support for JSON, YAML, Pickle, PyTorch, safetensors, and more
- **Flexible Dataset Management**: Organize data into splits (train/val/test) with associated assets
- **URI-based Identification**: Each shard has a unique URI for easy persistence and loading
- **Caching Support**: Optional in-memory caching for frequently accessed shards
- **Extensible**: Easy to add custom shard types and loaders

### Quick Example

```python
import tempfile
from pathlib import Path
from iden.dataset import create_vanilla_dataset
from iden.shard import create_json_shard, create_shard_dict, create_shard_tuple

# Create a simple dataset
with tempfile.TemporaryDirectory() as tmpdir:
    # Create shards
    train_tuple = create_shard_tuple(
        [
            create_json_shard(
                [1, 2, 3], uri=Path(tmpdir).joinpath("train1.json").as_uri()
            ),
            create_json_shard(
                [4, 5, 6], uri=Path(tmpdir).joinpath("train2.json").as_uri()
            ),
        ],
        uri=Path(tmpdir).joinpath("train_tuple").as_uri(),
    )
    val_tuple = create_shard_tuple(
        [create_json_shard([7, 8, 9], uri=Path(tmpdir).joinpath("val1.json").as_uri())],
        uri=Path(tmpdir).joinpath("val_tuple").as_uri(),
    )

    # Organize shards into splits
    shards = create_shard_dict(
        shards={"train": train_tuple, "val": val_tuple},
        uri=Path(tmpdir).joinpath("shards").as_uri(),
    )
    assets = create_shard_dict(shards={}, uri=Path(tmpdir).joinpath("assets").as_uri())

    # Create dataset
    dataset = create_vanilla_dataset(
        shards=shards,
        assets=assets,
        uri=Path(tmpdir).joinpath("my_dataset").as_uri(),
    )

    # Access data
    train_shards = dataset.get_shards("train")
    print(train_shards[0].get_data())  # Output: [1, 2, 3]
```

## API stability

:warning: While `iden` is in development stage, no API is guaranteed to be stable from one
release to the next. In fact, it is very likely that the API will change multiple times before a
stable 1.0.0 release. In practice, this means that upgrading `iden` to a new version will
possibly break any code that was using the old version of `iden`.

## License

`iden` is licensed under BSD 3-Clause "New" or "Revised" license available
in [LICENSE](https://github.com/durandtibo/iden/blob/main/LICENSE) file.
