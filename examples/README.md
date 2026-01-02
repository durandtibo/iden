# Examples

This directory contains practical examples demonstrating how to use the `iden` library.

## Available Examples

### 1. Basic Shard Usage (`basic_shard_usage.py`)

Learn the fundamentals of working with shards:
- Creating different types of shards (JSON, Pickle, YAML)
- Loading and saving data
- Using shard caching
- Loading shards from URIs

**Run:**
```bash
python examples/basic_shard_usage.py
```

### 2. Dataset Usage (`dataset_usage.py`)

Understand how to work with datasets:
- Creating datasets with multiple splits
- Accessing shards from datasets
- Working with assets (metadata, statistics)
- Saving and loading datasets
- Checking dataset properties

**Run:**
```bash
python examples/dataset_usage.py
```

### 3. I/O Operations (`io_operations.py`)

Explore various I/O operations:
- Saving and loading JSON, YAML, and Pickle files
- Handling complex Python objects
- Working with multiple files
- Error handling and best practices

**Run:**
```bash
python examples/io_operations.py
```

## Prerequisites

Make sure you have `iden` installed with the required dependencies:

```bash
# Install with all dependencies
pip install 'iden[all]'

# Or install with specific dependencies
pip install 'iden[yaml]'  # For YAML examples
```

## Learning Path

We recommend running the examples in this order:

1. **basic_shard_usage.py** - Start here to understand shards
2. **io_operations.py** - Learn about I/O operations
3. **dataset_usage.py** - Combine shards into datasets

## Contributing

If you have suggestions for new examples or improvements to existing ones, please open an issue or submit a pull request!
