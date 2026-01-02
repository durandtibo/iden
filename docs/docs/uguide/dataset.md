# Dataset

## Overview

The dataset is an abstraction to manage collections of shards organized into splits (e.g., train, validation, test) and assets (e.g., metadata, statistics).
`iden` provides a simple and flexible way to organize and access your machine learning datasets without loading all data into memory at once.

## Creating a dataset

A dataset can be created using the `create_vanilla_dataset` function or directly using the `VanillaDataset` class.

### Using create_vanilla_dataset

The `create_vanilla_dataset` function is the recommended way to create datasets:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.dataset import create_vanilla_dataset
>>> from iden.shard import create_json_shard, create_shard_dict, create_shard_tuple
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     # Create shards using create_shard_tuple
...     train_shards = create_shard_tuple(
...         [
...             create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("train1").as_uri()),
...             create_json_shard([4, 5, 6], uri=Path(tmpdir).joinpath("train2").as_uri()),
...         ],
...         uri=Path(tmpdir).joinpath("train_tuple").as_uri(),
...     )
...     # Create shards dictionary
...     shards = create_shard_dict(
...         shards={"train": train_shards},
...         uri=Path(tmpdir).joinpath("shards").as_uri(),
...     )
...     # Create assets (can be empty)
...     assets = create_shard_dict(
...         shards={},
...         uri=Path(tmpdir).joinpath("assets").as_uri(),
...     )
...     # Create dataset
...     dataset = create_vanilla_dataset(
...         shards=shards,
...         assets=assets,
...         uri=Path(tmpdir).joinpath("dataset").as_uri(),
...     )
...     dataset
...
VanillaDataset(
  (uri): file:///.../dataset
  (shards): ShardDict(...)
  (assets): ShardDict(...)
)

```

## Accessing shards

You can access shards by split name using the `get_shards` method, which returns a tuple of shards:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.dataset import create_vanilla_dataset
>>> from iden.shard import create_json_shard, create_shard_dict, create_shard_tuple
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     train_shards = create_shard_tuple(
...         [
...             create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("train1").as_uri()),
...             create_json_shard([4, 5, 6], uri=Path(tmpdir).joinpath("train2").as_uri()),
...         ],
...         uri=Path(tmpdir).joinpath("train_tuple").as_uri(),
...     )
...     val_shards = create_shard_tuple(
...         [create_json_shard([7, 8, 9], uri=Path(tmpdir).joinpath("val1").as_uri())],
...         uri=Path(tmpdir).joinpath("val_tuple").as_uri(),
...     )
...     shards = create_shard_dict(
...         shards={"train": train_shards, "val": val_shards},
...         uri=Path(tmpdir).joinpath("shards").as_uri(),
...     )
...     assets = create_shard_dict(
...         shards={},
...         uri=Path(tmpdir).joinpath("assets").as_uri(),
...     )
...     dataset = create_vanilla_dataset(
...         shards=shards,
...         assets=assets,
...         uri=Path(tmpdir).joinpath("dataset").as_uri(),
...     )
...     # Get training shards
...     train_data = dataset.get_shards("train")
...     # Access individual shards by index
...     first_shard = train_data[0]
...     first_shard.get_data()
...
[1, 2, 3]

```

## Working with assets

Assets allow you to store metadata, statistics, or other auxiliary information alongside your dataset:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.dataset import create_vanilla_dataset
>>> from iden.shard import create_json_shard, create_shard_dict, create_shard_tuple
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     # Create minimal shards
...     shards = create_shard_dict(
...         shards={
...             "train": create_shard_tuple(
...                 [
...                     create_json_shard(
...                         [1, 2, 3], uri=Path(tmpdir).joinpath("data").as_uri()
...                     )
...                 ],
...                 uri=Path(tmpdir).joinpath("train_tuple").as_uri(),
...             )
...         },
...         uri=Path(tmpdir).joinpath("shards").as_uri(),
...     )
...     # Create assets
...     assets = create_shard_dict(
...         shards={
...             "metadata": create_json_shard(
...                 {"version": "1.0", "num_samples": 1000},
...                 uri=Path(tmpdir).joinpath("metadata").as_uri(),
...             ),
...             "stats": create_json_shard(
...                 {"mean": 0.5, "std": 0.25},
...                 uri=Path(tmpdir).joinpath("stats").as_uri(),
...             ),
...         },
...         uri=Path(tmpdir).joinpath("assets").as_uri(),
...     )
...     dataset = create_vanilla_dataset(
...         shards=shards,
...         assets=assets,
...         uri=Path(tmpdir).joinpath("dataset").as_uri(),
...     )
...     # Access assets
...     metadata = dataset.get_asset("metadata")
...     metadata.get_data()
...
{'version': '1.0', 'num_samples': 1000}

```

## Available splits

You can check which splits are available in your dataset using `get_splits()`:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.dataset import create_vanilla_dataset
>>> from iden.shard import create_json_shard, create_shard_dict, create_shard_tuple
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     shards = create_shard_dict(
...         shards={
...             "train": create_shard_tuple(
...                 [
...                     create_json_shard(
...                         [1, 2, 3], uri=Path(tmpdir).joinpath("train").as_uri()
...                     )
...                 ],
...                 uri=Path(tmpdir).joinpath("train_tuple").as_uri(),
...             ),
...             "val": create_shard_tuple(
...                 [
...                     create_json_shard(
...                         [4, 5, 6], uri=Path(tmpdir).joinpath("val").as_uri()
...                     )
...                 ],
...                 uri=Path(tmpdir).joinpath("val_tuple").as_uri(),
...             ),
...             "test": create_shard_tuple(
...                 [
...                     create_json_shard(
...                         [7, 8, 9], uri=Path(tmpdir).joinpath("test").as_uri()
...                     )
...                 ],
...                 uri=Path(tmpdir).joinpath("test_tuple").as_uri(),
...             ),
...         },
...         uri=Path(tmpdir).joinpath("shards").as_uri(),
...     )
...     assets = create_shard_dict(shards={}, uri=Path(tmpdir).joinpath("assets").as_uri())
...     dataset = create_vanilla_dataset(
...         shards=shards,
...         assets=assets,
...         uri=Path(tmpdir).joinpath("dataset").as_uri(),
...     )
...     sorted(dataset.get_splits())
...
['test', 'train', 'val']

```

## Saving and loading datasets

Datasets are automatically saved when created with `create_vanilla_dataset` and can be loaded using `load_from_uri`:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.dataset import create_vanilla_dataset, load_from_uri
>>> from iden.shard import create_json_shard, create_shard_dict, create_shard_tuple
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     # Create and save dataset
...     shards = create_shard_dict(
...         shards={
...             "train": create_shard_tuple(
...                 [
...                     create_json_shard(
...                         [1, 2, 3], uri=Path(tmpdir).joinpath("train").as_uri()
...                     )
...                 ],
...                 uri=Path(tmpdir).joinpath("train_tuple").as_uri(),
...             )
...         },
...         uri=Path(tmpdir).joinpath("shards").as_uri(),
...     )
...     assets = create_shard_dict(shards={}, uri=Path(tmpdir).joinpath("assets").as_uri())
...     dataset = create_vanilla_dataset(
...         shards=shards,
...         assets=assets,
...         uri=Path(tmpdir).joinpath("dataset").as_uri(),
...     )
...     # Load dataset from URI
...     loaded_dataset = load_from_uri(Path(tmpdir).joinpath("dataset").as_uri())
...     loaded_dataset.get_uri()
...
'file:///.../dataset'

```
