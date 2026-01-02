# Troubleshooting

This page provides solutions to common issues you may encounter when using `iden`.

## Installation Issues

### Missing optional dependencies

**Problem**: You get an import error when trying to use certain features.

```python
ImportError: No module named 'torch'
```

**Solution**: Install the required optional dependency:

```shell
# For PyTorch support
pip install 'iden[torch]'

# For YAML support
pip install 'iden[pyyaml]'

# For safetensors support
pip install 'iden[safetensors]'

# Or install all optional dependencies
pip install 'iden[all]'
```

### Incompatible Python version

**Problem**: Installation fails due to Python version incompatibility.

**Solution**: Ensure you're using Python 3.10 or higher:

```shell
python --version  # Should be 3.10+
```

If needed, create a virtual environment with the correct Python version:

```shell
# Using conda
conda create -n iden python=3.10
conda activate iden

# Using uv
uv venv --python 3.10
```

## Shard Issues

### FileNotFoundError when loading shards

**Problem**: Getting `FileNotFoundError` when trying to load a shard.

```python
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/shard.json'
```

**Solution**: 

1. Verify the URI path exists:
   ```python
   from pathlib import Path
   path = Path("/path/to/shard.json")
   print(path.exists())
   ```

2. Ensure the shard was created before trying to load it:
   ```python
   from iden.shard import create_json_shard, load_from_uri
   
   # Create shard first
   shard = create_json_shard(data, uri=uri)
   
   # Then load it
   loaded_shard = load_from_uri(uri)
   ```

### Shard data not updating after modification

**Problem**: Changes to shard data are not persisted or appear stale.

**Solution**: 

1. Clear the cache before loading:
   ```python
   shard.clear()
   data = shard.get_data()
   ```

2. Don't use caching if you need fresh data:
   ```python
   # Without caching
   data = shard.get_data(cache=False)
   ```

### Memory issues with large datasets

**Problem**: Running out of memory when working with many shards.

**Solution**:

1. Avoid caching all shards:
   ```python
   # Don't do this for many shards
   for shard in shards:
       data = shard.get_data(cache=True)  # Memory leak!
   ```

2. Clear cache after processing:
   ```python
   for shard in shards:
       data = shard.get_data(cache=True)
       process(data)
       shard.clear()  # Free memory
   ```

3. Process shards in batches:
   ```python
   batch_size = 10
   for i in range(0, len(shards), batch_size):
       batch = shards[i:i+batch_size]
       # Process batch
       for shard in batch:
           shard.clear()
   ```

## Dataset Issues

### SplitNotFoundError

**Problem**: Accessing a split that doesn't exist.

```python
SplitNotFoundError: Split 'validation' not found
```

**Solution**:

1. Check available splits:
   ```python
   splits = dataset.get_splits()
   print(f"Available splits: {sorted(splits)}")
   ```

2. Use the correct split name:
   ```python
   # If the split is named 'val' not 'validation'
   shards = dataset.get_shards("val")
   ```

### AssetNotFoundError

**Problem**: Accessing an asset that doesn't exist.

```python
AssetNotFoundError: Asset 'statistics' not found
```

**Solution**:

1. Check if the asset exists:
   ```python
   if dataset.has_asset("statistics"):
       stats = dataset.get_asset("statistics")
   else:
       print("Asset not found")
   ```

## I/O Issues

### Permission denied when saving

**Problem**: Cannot save data due to permission issues.

```python
PermissionError: [Errno 13] Permission denied: '/path/to/file'
```

**Solution**:

1. Check directory permissions:
   ```shell
   ls -la /path/to/
   ```

2. Save to a writable location:
   ```python
   import tempfile
   from pathlib import Path
   
   # Use temporary directory
   with tempfile.TemporaryDirectory() as tmpdir:
       path = Path(tmpdir).joinpath("data.json")
       save_json(data, path)
   ```

### Unsupported file format

**Problem**: Trying to use a format without the required dependency.

**Solution**: Install the required package:

```shell
# For YAML files
pip install pyyaml

# For PyTorch files
pip install torch

# For safetensors
pip install safetensors
```

## Performance Issues

### Slow shard loading

**Problem**: Loading shards is slower than expected.

**Solution**:

1. Use caching for frequently accessed shards:
   ```python
   data = shard.get_data(cache=True)
   ```

2. Use more efficient formats:
   - Use `safetensors` instead of `pickle` for tensors
   - Use `cloudpickle` for complex Python objects
   - Avoid `json` for large numerical arrays

3. Parallelize loading:
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor() as executor:
       futures = [executor.submit(shard.get_data) for shard in shards]
       data = [f.result() for f in futures]
   ```

### High memory usage

**Problem**: Application using too much memory.

**Solution**:

1. Use lazy loading (don't cache unnecessarily):
   ```python
   # Good - lazy loading
   shard = create_json_shard(data, uri=uri)
   
   # Bad - immediate caching
   shard = create_json_shard(data, uri=uri)
   shard.get_data(cache=True)  # Loads into memory immediately
   ```

2. Process data in smaller batches
3. Clear caches regularly with `shard.clear()`

## Getting Help

If you continue to experience issues:

1. Check the [documentation](https://durandtibo.github.io/iden/)
2. Search [existing issues](https://github.com/durandtibo/iden/issues)
3. Ask a question in [GitHub Discussions](https://github.com/durandtibo/iden/discussions)
4. Open a [new issue](https://github.com/durandtibo/iden/issues/new) with:
   - Your Python version
   - Your `iden` version
   - A minimal code example to reproduce the issue
   - The full error traceback
