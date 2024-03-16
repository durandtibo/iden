# Shard

## Overview

The shard is an abstraction to represent a unit of data.
It provides an abstraction to get the data without knowing how the data are stored.
Each shard must have a unique Uniform Resource Identifier (URI), which is used to identify each
shard, so it is possible to instantiate a shard from its URI.
The `get_uri` method can be used to get the URI of shard:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.shard import create_json_shard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     shard = create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("my_uri").as_uri())
...     uri = shard.get_uri()
...     uri
...
'file:///.../my_uri'

```

To be scalable, a shard does not contain the data, but it contains the logic to get the data.
It allows to create and manage a large number of shards independently of the total size of the data.
The `get_data` method is used to get the data from the shard:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.shard import create_json_shard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     shard = create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("my_uri").as_uri())
...     data = shard.get_data()
...     data
...
[1, 2, 3]

```

Most of the shard caches the data in-memory after the data are loaded the first time.
It is possible to clear the cache by calling the `clear` method.

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.shard import create_json_shard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     shard = create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("my_uri").as_uri())
...     data = shard.get_data(cache=True)
...     data
...     data.append(4)  # in-place modification
...     data = shard.get_data()
...     data
...     shard.clear()
...     data = shard.get_data()
...     data
...
...
[1, 2, 3]
[1, 2, 3, 4]
[1, 2, 3]

```

It is important to clear the `cache` if the shard is not used because it can lead to OOM issues if
the data of too may shards are cached in-memory at the same time.
It is possible to call the `is_initialized` method to know if the data in the data are cached or
not.

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.shard import create_json_shard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     shard = create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("my_uri").as_uri())
...     shard.is_initialized()
...     data = shard.get_data(cache=True)
...     shard.is_initialized()
...     shard.clear()
...     shard.is_initialized()
...
...
False
True
False

```

Finally, there is the `equal` method to check if two shards are equal or not:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.shard import JsonShard,  create_json_shard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     uri1 = Path(tmpdir).joinpath("my_uri1").as_uri()
...     uri2 = Path(tmpdir).joinpath("my_uri2").as_uri()
...     shard1 = create_json_shard([1, 2, 3], uri=uri1)
...     shard2 = create_json_shard([4, 5, 6], uri=uri2)
...     shard3 = JsonShard.from_uri(uri=uri1)
...     shard1.equal(shard2)
...     shard1.equal(shard3)
...
...
False
True

```

## Built-in shards

`iden` has some built-in shard implementations that can be used out of the box.
It is possible to extend `iden` to support more shard implementation.
[This page](../howto/shard.md) explains how to add a new shard implementation.

Each shard implementation is different and has different properties.
You need to choose the best shard based on your requirements. It is not a one size fits all.
For example, the `PickleShard` implementation supports a lot of types of data whereas
the `TorchSafetensorsShard` implementation only supports dictionary of ``torch.Tensor``s.
The following table shows a summary of supported data for some of the built-in shards.

| shard                   | supported data                       |
|-------------------------|--------------------------------------|
| `FileShard`             | depend on the file format            |
| `JsonShard`             | any data compatible with JSON format |
| `PickleShard`           | any serializable data                |
| `TorchSafetensorsShard` | a dictionary of ``torch.Tensor``s    |
| `TorchShard`            | any serializable data                |
| `YamlShard`             | any data compatible with YAML format |

**File-based shards.**
`iden` has some shard implementations to load data from files.
`iden` relies on existing packages to save and load data in s shard.
Most of these packages are optional and should be installed if necessary.
The following table shows some of the supported file format, the package used to save and load data,
and their associated shard implementations.

| shard                   | file format                                                               | package       |
|-------------------------|---------------------------------------------------------------------------|---------------|
| `JsonShard`             | [JSON file](https://docs.python.org/3/library/json.html)                  | `json`        |
| `PickleShard`           | [pickle file](https://docs.python.org/3/library/pickle.html)              | `yaml`        |
| `TorchSafetensorsShard` | [safetensors file](https://huggingface.co/docs/safetensors/en/index)      | `safetensors` |
| `TorchShard`            | [pytorch file](https://pytorch.org/docs/stable/generated/torch.save.html) | `torch`       |
| `YamlShard`             | [YAML file](https://pyyaml.org/)                                          | `yaml`        |

`FileShard` is generic file-based shard that supports most of the file formats.

**Special shards.**
`iden` has some special shards that allows to combine multiple shards.
`ShardTuple` is the shard implementation to manage a tuple of shards.

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.shard import ShardTuple, create_json_shard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     shards = [
...         create_json_shard([1, 2, 3], uri=Path(tmpdir).joinpath("shards/uri1").as_uri()),
...         create_json_shard(
...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shards/uri2").as_uri()
...         ),
...     ]
...     sl = ShardTuple(uri=Path(tmpdir).joinpath("uri").as_uri(), shards=shards)
...     sl
...
ShardTuple(
  (uri): file:///.../uri
  (shards):
    (0): JsonShard(uri=file:///.../shards/uri1)
    (1): JsonShard(uri=file:///.../shards/uri2)
)

```

`ShardDict` is the shard implementation to manage a dictionary of shards.

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.shard import ShardDict, create_json_shard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     shards = {
...         "train": create_json_shard(
...             [1, 2, 3], uri=Path(tmpdir).joinpath("shards/uri1").as_uri()
...         ),
...         "val": create_json_shard(
...             [4, 5, 6, 7], uri=Path(tmpdir).joinpath("shards/uri2").as_uri()
...         ),
...     }
...     sd = ShardDict(uri=Path(tmpdir).joinpath("uri").as_uri(), shards=shards)
...     sd
...
ShardDict(
  (uri): file:///.../uri
  (shards):
    (train): JsonShard(uri=file:///.../shards/uri1)
    (val): JsonShard(uri=file:///.../shards/uri2)
)

```

## Instantiating a shard from its URI

`iden` has a functionality to instantiate a shard from its Uniform Resource Identifier (URI).
A shard can be represented by its URI, which can help to make the data management more scalable.
It is easier to manage the URIs than all the data.
The `load_from_uri` function can be used to instantiate a shard from its URI.

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.shard import create_json_shard, load_from_uri
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     uri = Path(tmpdir).joinpath("my_uri").as_uri()
...     _ = create_json_shard([1, 2, 3], uri=uri)
...     shard = load_from_uri(uri)
...     shard
...
JsonShard(uri=file:///.../my_uri)

```

Under the hood, the `load_from_uri` function relies on a shard loader object to instantiate a shard
object.
A shard loader contains the logic to instantiate a shard object from its URI.
For instance in the previous example, the `JsonShardLoader` class is used to instantiate
the `JsonShard` object.
`load_from_uri` is a universal function to load any shards, but it is also possible to use specific
data loaders.
For instance, the following example is equivalent to the previous example:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.shard import create_json_shard
>>> from iden.shard.loader import JsonShardLoader
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     uri = Path(tmpdir).joinpath("my_uri").as_uri()
...     _ = create_json_shard([1, 2, 3], uri=uri)
...     loader = JsonShardLoader()
...     shard = loader.load(uri)
...     shard
...
JsonShard(uri=file:///.../my_uri)

```

## Uniform Resource Identifier (URI)

The URI file contains enough information to instantiate the shard object, and it is encoded as a
JSON file.
All URI files should contain a dictionary, with at least one key which indicates which the shard
loader to use to instantiate the shard.
The following example shows the generate configuration for a `JsonShard` object:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from iden.shard import JsonShard
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("data.json")
...     config = JsonShard.generate_uri_config(path)
...     config
...
{'kwargs': {'path': '/.../data.json'},
 'loader': {'_target_': 'iden.shard.loader.JsonShardLoader'}}

```

The `'kwargs'` key is specific to `JsonShard` and indicates where to find the JSON file associated
to the shard.
