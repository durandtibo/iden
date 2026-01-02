"""Example: Working with datasets.

This example demonstrates how to create and use datasets in iden.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from iden.dataset import create_vanilla_dataset, load_from_uri
from iden.shard import create_json_shard


def main() -> None:
    """Run the dataset example."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Example 1: Creating a simple dataset
        print("=" * 60)
        print("Example 1: Creating a simple dataset")
        print("=" * 60)
        
        dataset = create_vanilla_dataset(
            uri=base_path.joinpath("dataset").as_uri(),
            shards={
                "train": [
                    create_json_shard([1, 2, 3, 4, 5], uri=base_path.joinpath("train1.json").as_uri()),
                    create_json_shard([6, 7, 8, 9, 10], uri=base_path.joinpath("train2.json").as_uri()),
                ],
                "val": [
                    create_json_shard([11, 12, 13], uri=base_path.joinpath("val1.json").as_uri()),
                ],
                "test": [
                    create_json_shard([14, 15, 16], uri=base_path.joinpath("test1.json").as_uri()),
                ],
            },
        )
        
        print(f"Dataset URI: {dataset.get_uri()}")
        print(f"Available splits: {dataset.get_split_names()}")
        print()

        # Example 2: Accessing shards
        print("=" * 60)
        print("Example 2: Accessing shards from dataset")
        print("=" * 60)
        
        train_shards = dataset.get_shards("train")
        print(f"Train shards: {train_shards}")
        print(f"Number of train shards: {dataset.get_num_shards('train')}")
        
        # Get individual shard
        first_train_shard = dataset.get_shard("train", 0)
        print(f"First train shard data: {first_train_shard.get_data()}")
        print()

        # Example 3: Working with assets
        print("=" * 60)
        print("Example 3: Dataset with assets")
        print("=" * 60)
        
        dataset_with_assets = create_vanilla_dataset(
            uri=base_path.joinpath("dataset_with_assets").as_uri(),
            shards={
                "train": [
                    create_json_shard([1, 2, 3], uri=base_path.joinpath("train_data.json").as_uri()),
                ],
            },
            assets={
                "metadata": create_json_shard(
                    {"version": "1.0.0", "created_by": "example", "num_samples": 3},
                    uri=base_path.joinpath("metadata.json").as_uri(),
                ),
                "statistics": create_json_shard(
                    {"mean": 2.0, "std": 0.816, "min": 1, "max": 3},
                    uri=base_path.joinpath("statistics.json").as_uri(),
                ),
            },
        )
        
        print(f"Available assets: {dataset_with_assets.get_asset_names()}")
        
        # Access asset
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
        original_dataset = create_vanilla_dataset(
            uri=base_path.joinpath("persistent_dataset").as_uri(),
            shards={
                "train": [
                    create_json_shard(
                        {"samples": list(range(100))},
                        uri=base_path.joinpath("persistent_train.json").as_uri(),
                    ),
                ],
            },
        )
        
        print(f"Original dataset URI: {original_dataset.get_uri()}")
        
        # Load the dataset from URI
        loaded_dataset = load_from_uri(base_path.joinpath("persistent_dataset").as_uri())
        print(f"Loaded dataset URI: {loaded_dataset.get_uri()}")
        print(f"Loaded dataset splits: {loaded_dataset.get_split_names()}")
        
        # Verify data integrity
        original_data = original_dataset.get_shard("train", 0).get_data()
        loaded_data = loaded_dataset.get_shard("train", 0).get_data()
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
