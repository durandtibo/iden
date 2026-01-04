# Frequently Asked Questions (FAQ)

## General Questions

### What is `iden`?

`iden` is a Python library for managing datasets of shards when training machine learning models. It
provides a lazy loading approach to handle large datasets efficiently without loading everything
into memory at once.

### Why use `iden`?

- **Memory Efficient**: Load data only when needed using lazy loading
- **Scalable**: Manage large datasets that don't fit in memory
- **Flexible**: Support for multiple data formats (JSON, YAML, Pickle, PyTorch, etc.)
- **Organized**: Split datasets into train/val/test with associated metadata
- **Easy to Use**: Simple API for creating, loading, and managing shards
-

## Installation Questions

### What Python versions are supported?

`iden` requires Python 3.10 or higher.

### Do I need to install all dependencies?

No! `iden` has a minimal installation that only includes core dependencies.
Optional dependencies can be installed as needed.

For detailed installation instructions and examples, please refer to the [Get Started guide](https://durandtibo.github.io/iden/get_started).

### Can I use `iden` with my existing PyTorch/TensorFlow workflow?

Yes! `iden` is framework-agnostic. You can store PyTorch tensors, TensorFlow tensors, NumPy arrays,
or any serializable Python objects.

## Usage Questions

### What's the difference between a shard and a dataset?

- **Shard**: A single unit of data (e.g., one file containing samples)
- **Dataset**: A collection of shards organized into splits (train/val/test) with optional assets (
  metadata, statistics)

### How do I choose the right shard format?

Consider these factors:

| Format      | Best For             | Pros                         | Cons                |
|-------------|----------------------|------------------------------|---------------------|
| JSON        | Simple data, configs | Human-readable, universal    | Slow for large data |
| Pickle      | Python objects       | Supports any Python object   | Not cross-language  |
| PyTorch     | PyTorch tensors      | Fast, PyTorch integration    | Requires PyTorch    |
| safetensors | Large tensors        | Fast, safe, efficient        | Tensors only        |
| YAML        | Configurations       | Human-readable, config files | Slow for large data |

### Should I cache shard data?

Use caching when:

- The shard is accessed multiple times
- The data is small enough to fit in memory
- Loading is slow (e.g., from network storage)

Don't cache when:

- You have many shards (risk of OOM)
- The data is only used once
- Memory is limited

### How do I handle very large datasets?

1. **Use sharding**: Split data into multiple small shards
2. **Lazy loading**: Don't cache unless necessary
3. **Process in batches**: Load and process shards in small batches
4. **Efficient formats**: Use `safetensors` for tensors, avoid JSON for large arrays
5. **Clear cache**: Call `shard.clear()` after processing

Example:

```python
for i in range(0, len(shards), batch_size):
    batch_shards = shards[i : i + batch_size]
    for shard in batch_shards:
        data = shard.get_data()
        process(data)
        shard.clear()  # Free memory
```

## Technical Questions

### What is a URI in `iden`?

A URI (Uniform Resource Identifier) is a unique identifier for each shard or dataset.
It's typically a file path converted to a URI format:

```python
from pathlib import Path

uri = Path("/path/to/data.json").as_uri()
# Result: "file:///path/to/data.json"
```

### Can I use `iden` with remote storage (S3, GCS, etc.)?

Currently, `iden` primarily supports local file systems. However, you can:

1. Download data locally first, then use `iden`
2. Use file-like objects with compatible loaders
3. Extend `iden` with custom loaders for remote storage

### Can I extend `iden` with custom shard types?

Yes! Implement a custom shard by extending `BaseShard`:

```python
from iden.shard import BaseShard


class CustomShard(BaseShard):
    def get_data(self, cache: bool = False):
        # Your implementation
        pass

    def clear(self):
        # Your implementation
        pass

    # Implement other required methods...
```

## Performance Questions

### Why is loading slow?

Possible reasons:

1. Using inefficient format (e.g., JSON for large arrays)
2. Loading from slow storage (network drive, slow disk)
3. Large files without compression
4. Not using caching for repeated access

### How much memory does `iden` use?

`iden` itself is lightweight. Memory usage depends on:

- How many shards you cache
- The size of your data
- Whether you use lazy loading properly

## Contributing Questions

### How can I contribute?

Please refer to the [CONTRIBUTING.md](https://github.com/durandtibo/iden/blob/main/CONTRIBUTING.md) guide for detailed instructions on:

- Setting up your development environment
- Running tests
- Code quality checks
- Submitting pull requests

### I found a bug, what should I do?

1. Check if it's already reported in [issues](https://github.com/durandtibo/iden/issues)
2. If not, open a new issue with:
    - Python version
    - `iden` version
    - Minimal code to reproduce
    - Full error traceback

### Can I add support for a new file format?

Yes! We welcome contributions. Open an issue to discuss, then submit a PR with:

- New shard class
- New loader class
- Tests
- Documentation
- Example usage

## Getting Help

### Where can I get help?

1. Read the [documentation](https://durandtibo.github.io/iden/)
2. Check this FAQ
3. Search [existing issues](https://github.com/durandtibo/iden/issues)
4. Ask in [GitHub Discussions](https://github.com/durandtibo/iden/discussions)
5. Open a [new issue](https://github.com/durandtibo/iden/issues/new)

### How do I report a security vulnerability?

Please email the maintainers privately rather than opening a public issue.
See [SECURITY.md](https://github.com/durandtibo/iden/blob/main/SECURITY.md) if it exists, or contact
through GitHub.
