"""Example: Basic shard usage.

This example demonstrates how to create and use different types of shards
in iden.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from iden.shard import (
    create_json_shard,
    create_pickle_shard,
    create_yaml_shard,
    load_from_uri,
)


def main() -> None:
    """Run the basic shard example."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Example 1: JSON shard
        print("=" * 60)
        print("Example 1: Creating and using a JSON shard")
        print("=" * 60)
        
        data = {"name": "Alice", "age": 30, "hobbies": ["reading", "coding"]}
        json_uri = base_path.joinpath("person.json").as_uri()
        
        # Create shard
        json_shard = create_json_shard(data, uri=json_uri)
        print(f"Created JSON shard with URI: {json_shard.get_uri()}")
        
        # Get data
        loaded_data = json_shard.get_data()
        print(f"Loaded data: {loaded_data}")
        
        # Load from URI
        shard_from_uri = load_from_uri(json_uri)
        print(f"Loaded shard from URI: {shard_from_uri}")
        print()

        # Example 2: Pickle shard with complex data
        print("=" * 60)
        print("Example 2: Creating a Pickle shard with complex data")
        print("=" * 60)
        
        complex_data = {
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "set": {1, 2, 3},
            "tuple": (1, 2, 3),
        }
        pickle_uri = base_path.joinpath("complex_data.pkl").as_uri()
        
        pickle_shard = create_pickle_shard(complex_data, uri=pickle_uri)
        print(f"Created Pickle shard with URI: {pickle_shard.get_uri()}")
        
        loaded_complex = pickle_shard.get_data()
        print(f"Loaded complex data: {loaded_complex}")
        print()

        # Example 3: YAML shard
        print("=" * 60)
        print("Example 3: Creating a YAML shard")
        print("=" * 60)
        
        config = {
            "model": {"name": "resnet50", "pretrained": True},
            "training": {"epochs": 100, "batch_size": 32, "lr": 0.001},
        }
        yaml_uri = base_path.joinpath("config.yaml").as_uri()
        
        yaml_shard = create_yaml_shard(config, uri=yaml_uri)
        print(f"Created YAML shard with URI: {yaml_shard.get_uri()}")
        
        loaded_config = yaml_shard.get_data()
        print(f"Loaded config: {loaded_config}")
        print()

        # Example 4: Caching
        print("=" * 60)
        print("Example 4: Using shard caching")
        print("=" * 60)
        
        data = list(range(1000))
        cached_uri = base_path.joinpath("large_data.json").as_uri()
        cached_shard = create_json_shard(data, uri=cached_uri)
        
        print(f"Is cached initially? {cached_shard.is_cached()}")
        
        # Load with caching
        _ = cached_shard.get_data(cache=True)
        print(f"Is cached after loading? {cached_shard.is_cached()}")
        
        # Clear cache
        cached_shard.clear()
        print(f"Is cached after clearing? {cached_shard.is_cached()}")
        print()

        print("=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    main()
