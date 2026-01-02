"""Example: Working with datasets.

This example demonstrates how to create and use datasets in iden.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from iden.dataset import VanillaDataset, create_vanilla_dataset, load_from_uri
from iden.shard import create_json_shard, create_shard_dict, create_shard_tuple


def main() -> None:
    """Run the dataset example."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Example 1: Creating a simple dataset
        print("=" * 60)
        print("Example 1: Creating a simple dataset")
        print("=" * 60)
        
        # Create shards using create_shard_dict and create_shard_tuple
        shards = create_shard_dict(
            shards={
                "train": create_shard_tuple(
                    [
                        create_json_shard([1, 2, 3, 4, 5], uri=base_path.joinpath("train1.json").as_uri()),
                        create_json_shard([6, 7, 8, 9, 10], uri=base_path.joinpath("train2.json").as_uri()),
                    ],
                    uri=base_path.joinpath("train_shards").as_uri(),
                ),
                "val": create_shard_tuple(
                    [create_json_shard([11, 12, 13], uri=base_path.joinpath("val1.json").as_uri())],
                    uri=base_path.joinpath("val_shards").as_uri(),
                ),
            },
            uri=base_path.joinpath("all_shards").as_uri(),
        )
        
        assets = create_shard_dict(
            shards={},
            uri=base_path.joinpath("assets").as_uri(),
        )
        
        dataset = create_vanilla_dataset(
            shards=shards,
            assets=assets,
            uri=base_path.joinpath("dataset").as_uri(),
        )
        
        print(f"Dataset URI: {dataset.get_uri()}")
        print(f"Available splits: {sorted(dataset.get_splits())}")
        print()

        # Example 2: Accessing shards
        print("=" * 60)
        print("Example 2: Accessing shards from dataset")
        print("=" * 60)
        
        train_shards = dataset.get_shards("train")
        print(f"Number of train shards: {dataset.get_num_shards('train')}")
        
        # Get data from first shard
        first_train_data = train_shards[0].get_data()
        print(f"First train shard data: {first_train_data}")
        print()

        # Example 3: Working with assets
        print("=" * 60)
        print("Example 3: Dataset with assets")
        print("=" * 60)
        
        shards_with_assets = create_shard_dict(
            shards={
                "train": create_shard_tuple(
                    [create_json_shard([1, 2, 3], uri=base_path.joinpath("train_data.json").as_uri())],
                    uri=base_path.joinpath("train_shards_wa").as_uri(),
                ),
            },
            uri=base_path.joinpath("all_shards_wa").as_uri(),
        )
        
        assets_with_data = create_shard_dict(
            shards={
                "metadata": create_json_shard(
                    {"version": "1.0.0", "created_by": "example", "num_samples": 3},
                    uri=base_path.joinpath("metadata.json").as_uri(),
                ),
                "statistics": create_json_shard(
                    {"mean": 2.0, "std": 0.816, "min": 1, "max": 3},
                    uri=base_path.joinpath("statistics.json").as_uri(),
                ),
            },
            uri=base_path.joinpath("assets_wa").as_uri(),
        )
        
        dataset_with_assets = create_vanilla_dataset(
            shards=shards_with_assets,
            assets=assets_with_data,
            uri=base_path.joinpath("dataset_with_assets").as_uri(),
        )
        
        # Access assets
        print(f"Has 'metadata' asset: {dataset_with_assets.has_asset('metadata')}")
        
        metadata = dataset_with_assets.get_asset("metadata")
        print(f"Metadata: {metadata.get_data()}")
        
        stats = dataset_with_assets.get_asset("statistics")
        print(f"Statistics: {stats.get_data()}")
        print()

        # Example 4: Saving and loading datasets
        print("=" * 60)
        print("Example 4: Saving and loading datasets")
        print("=" * 60)
        
        # Create a dataset
        persistent_shards = create_shard_dict(
            shards={
                "train": create_shard_tuple(
                    [
                        create_json_shard(
                            {"samples": list(range(100))},
                            uri=base_path.joinpath("persistent_train.json").as_uri(),
                        ),
                    ],
                    uri=base_path.joinpath("persistent_train_shards").as_uri(),
                ),
            },
            uri=base_path.joinpath("persistent_all_shards").as_uri(),
        )
        
        persistent_assets = create_shard_dict(
            shards={},
            uri=base_path.joinpath("persistent_assets").as_uri(),
        )
        
        original_dataset = create_vanilla_dataset(
            shards=persistent_shards,
            assets=persistent_assets,
            uri=base_path.joinpath("persistent_dataset").as_uri(),
        )
        
        print(f"Original dataset URI: {original_dataset.get_uri()}")
        
        # Load the dataset from URI
        loaded_dataset = load_from_uri(base_path.joinpath("persistent_dataset").as_uri())
        print(f"Loaded dataset URI: {loaded_dataset.get_uri()}")
        print(f"Loaded dataset splits: {sorted(loaded_dataset.get_splits())}")
        
        # Verify data integrity
        original_data = original_dataset.get_shards("train")[0].get_data()
        loaded_data = loaded_dataset.get_shards("train")[0].get_data()
        print(f"Data matches: {original_data == loaded_data}")
        print()

        # Example 5: Checking dataset properties
        print("=" * 60)
        print("Example 5: Checking dataset properties")
        print("=" * 60)
        
        print(f"Has 'train' split: {dataset.has_split('train')}")
        print(f"Has 'invalid' split: {dataset.has_split('invalid')}")
        print(f"Number of validation shards: {dataset.get_num_shards('val')}")
        print()

        print("=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    main()
