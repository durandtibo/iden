# Dataset

## Overview

The dataset is an abstraction to manage collections of shards organized into splits (e.g., train, validation, test) and assets (e.g., metadata, statistics).
`iden` provides a simple and flexible way to organize and access your machine learning datasets without loading all data into memory at once.

## Creating a dataset

A dataset can be created using the `VanillaDataset` class, which requires:

- **uri**: A unique identifier for the dataset
- **shards**: A dictionary of shards organized by split names
- **assets**: A dictionary of assets (optional)

Here's a simple example:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.dataset import VanillaDataset
>>> from iden.shard import create_json_shard, create_shard_dict, create_shard_tuple
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     # Create shards for training data
...     train_shards = create_shard_tuple(
...         [
...             create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("train/shard1").as_uri()),
...             create_json_shard([4, 5, 6], uri=Path(tmpdir).joinpath("train/shard2").as_uri()),
...         ],
...         uri=Path(tmpdir).joinpath("train_shards").as_uri(),
...     )
...     # Create shards dictionary
...     shards = create_shard_dict(
...         shards={"train": train_shards},
...         uri=Path(tmpdir).joinpath("shards").as_uri(),
...     )
...     # Create dataset
...     dataset = VanillaDataset(
...         uri=Path(tmpdir).joinpath("dataset").as_uri(),
...         shards=shards,
...     )
...     dataset
...
VanillaDataset(
  (uri): file:///.../dataset
  (shards): ShardDict(
      (uri): file:///.../shards
      (shards):
        (train): ShardTuple(
            (uri): file:///.../train_shards
            (shards):
              (0): JsonShard(uri=file:///.../train/shard1)
              (1): JsonShard(uri=file:///.../train/shard2)
          )
    )
  (assets): ShardDict(...)
)

```

## Accessing shards

You can access shards by split name using the `get_shards` method:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.dataset import create_vanilla_dataset
>>> from iden.shard import create_json_shard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     dataset = create_vanilla_dataset(
...         uri=Path(tmpdir).joinpath("dataset").as_uri(),
...         shards={
...             "train": [
...                 create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("train1").as_uri()),
...                 create_json_shard([4, 5, 6], uri=Path(tmpdir).joinpath("train2").as_uri()),
...             ],
...             "val": [
...                 create_json_shard([7, 8, 9], uri=Path(tmpdir).joinpath("val1").as_uri()),
...             ],
...         },
...     )
...     # Get training shards
...     train_shards = dataset.get_shards("train")
...     train_shards
...
ShardTuple(
  (uri): file:///.../dataset/shards/train
  (shards):
    (0): JsonShard(uri=file:///.../train1)
    (1): JsonShard(uri=file:///.../train2)
)

```

## Working with assets

Assets allow you to store metadata, statistics, or other auxiliary information alongside your dataset:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.dataset import VanillaDataset
>>> from iden.shard import create_json_shard, create_shard_dict
>>> with tempfile.TemporaryDirectory() as tmpdir:
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
...     dataset = VanillaDataset(
...         uri=Path(tmpdir).joinpath("dataset").as_uri(),
...         shards=create_shard_dict(shards={}, uri=Path(tmpdir).joinpath("shards").as_uri()),
...         assets=assets,
...     )
...     # Access assets
...     metadata = dataset.get_asset("metadata")
...     metadata.get_data()
...
{'version': '1.0', 'num_samples': 1000}

```

## Available splits

You can check which splits are available in your dataset:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.dataset import create_vanilla_dataset
>>> from iden.shard import create_json_shard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     dataset = create_vanilla_dataset(
...         uri=Path(tmpdir).joinpath("dataset").as_uri(),
...         shards={
...             "train": [create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("train").as_uri())],
...             "val": [create_json_shard([4, 5, 6], uri=Path(tmpdir).joinpath("val").as_uri())],
...             "test": [create_json_shard([7, 8, 9], uri=Path(tmpdir).joinpath("test").as_uri())],
...         },
...     )
...     dataset.get_split_names()
...
('test', 'train', 'val')

```

## Saving and loading datasets

Datasets can be saved to disk and loaded back:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.dataset import create_vanilla_dataset, load_from_uri
>>> from iden.shard import create_json_shard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     # Create and save dataset
...     dataset = create_vanilla_dataset(
...         uri=Path(tmpdir).joinpath("dataset").as_uri(),
...         shards={
...             "train": [create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("train").as_uri())],
...         },
...     )
...     # Load dataset from URI
...     loaded_dataset = load_from_uri(Path(tmpdir).joinpath("dataset").as_uri())
...     loaded_dataset
...
VanillaDataset(
  (uri): file:///.../dataset
  (shards): ShardDict(...)
  (assets): ShardDict(...)
)

```

## Helper function

The `create_vanilla_dataset` function provides a convenient way to create datasets:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.dataset import create_vanilla_dataset
>>> from iden.shard import create_json_shard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     dataset = create_vanilla_dataset(
...         uri=Path(tmpdir).joinpath("dataset").as_uri(),
...         shards={
...             "train": [
...                 create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("train1").as_uri()),
...             ],
...         },
...         assets={
...             "metadata": create_json_shard(
...                 {"version": "1.0"},
...                 uri=Path(tmpdir).joinpath("metadata").as_uri(),
...             ),
...         },
...     )
...     dataset
...
VanillaDataset(
  (uri): file:///.../dataset
  (shards): ShardDict(...)
  (assets): ShardDict(...)
)

```
