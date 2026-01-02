"""Example: I/O operations with different formats.

This example demonstrates how to save and load data in various formats.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from iden.io import (
    load_json,
    load_pickle,
    load_yaml,
    save_json,
    save_pickle,
    save_yaml,
)


def main() -> None:
    """Run the I/O example."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Example 1: JSON I/O
        print("=" * 60)
        print("Example 1: JSON format")
        print("=" * 60)
        
        json_data = {
            "model": "transformer",
            "config": {
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.0001,
            },
        }
        
        json_path = base_path.joinpath("config.json")
        save_json(json_data, json_path)
        print(f"Saved JSON to: {json_path}")
        
        loaded_json = load_json(json_path)
        print(f"Loaded JSON: {loaded_json}")
        print(f"Data matches: {json_data == loaded_json}")
        print()

        # Example 2: YAML I/O
        print("=" * 60)
        print("Example 2: YAML format")
        print("=" * 60)
        
        yaml_data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "mydb",
            },
            "features": ["feature1", "feature2", "feature3"],
            "enabled": True,
        }
        
        yaml_path = base_path.joinpath("config.yaml")
        save_yaml(yaml_data, yaml_path)
        print(f"Saved YAML to: {yaml_path}")
        
        loaded_yaml = load_yaml(yaml_path)
        print(f"Loaded YAML: {loaded_yaml}")
        print(f"Data matches: {yaml_data == loaded_yaml}")
        print()

        # Example 3: Pickle I/O (supports complex Python objects)
        print("=" * 60)
        print("Example 3: Pickle format (complex objects)")
        print("=" * 60)
        
        class CustomObject:
            def __init__(self, name: str, value: int) -> None:
                self.name = name
                self.value = value
            
            def __eq__(self, other: object) -> bool:
                if not isinstance(other, CustomObject):
                    return False
                return self.name == other.name and self.value == other.value
            
            def __repr__(self) -> str:
                return f"CustomObject(name={self.name!r}, value={self.value})"
        
        pickle_data = {
            "objects": [
                CustomObject("obj1", 10),
                CustomObject("obj2", 20),
            ],
            "set": {1, 2, 3, 4, 5},
            "tuple": (1, 2, 3),
        }
        
        pickle_path = base_path.joinpath("data.pkl")
        save_pickle(pickle_data, pickle_path)
        print(f"Saved Pickle to: {pickle_path}")
        
        loaded_pickle = load_pickle(pickle_path)
        print(f"Loaded Pickle objects: {loaded_pickle['objects']}")
        print(f"Data matches: {pickle_data == loaded_pickle}")
        print()

        # Example 4: Multiple files
        print("=" * 60)
        print("Example 4: Working with multiple files")
        print("=" * 60)
        
        # Save multiple configuration files
        configs = {
            "model_config": {"type": "cnn", "layers": 5},
            "data_config": {"batch_size": 64, "num_workers": 4},
            "optimizer_config": {"type": "adam", "lr": 0.001},
        }
        
        for name, config in configs.items():
            path = base_path.joinpath(f"{name}.json")
            save_json(config, path)
            print(f"Saved {name} to: {path}")
        
        # Load all configurations
        loaded_configs = {}
        for name in configs:
            path = base_path.joinpath(f"{name}.json")
            loaded_configs[name] = load_json(path)
        
        print(f"\nAll configs match: {configs == loaded_configs}")
        print()

        # Example 5: Error handling
        print("=" * 60)
        print("Example 5: Error handling")
        print("=" * 60)
        
        try:
            # Try to load a non-existent file
            load_json(base_path.joinpath("nonexistent.json"))
        except FileNotFoundError as e:
            print(f"Caught expected error: {e.__class__.__name__}")
        
        # Save to non-existent directory (should auto-create)
        deep_path = base_path.joinpath("a/b/c/config.json")
        save_json({"test": "data"}, deep_path)
        print(f"Successfully saved to nested path: {deep_path}")
        print(f"File exists: {deep_path.exists()}")
        print()

        print("=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    main()
