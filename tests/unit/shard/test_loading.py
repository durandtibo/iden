from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from coola import objects_are_equal
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available
from coola.utils.path import sanitize_path

from iden.io import JsonSaver
from iden.shard import (
    CloudpickleShard,
    FileShard,
    JsonShard,
    NumpySafetensorsShard,
    PickleShard,
    ShardTuple,
    TorchSafetensorsShard,
    TorchShard,
    YamlShard,
    create_cloudpickle_shard,
    create_json_shard,
    create_numpy_safetensors_shard,
    create_pickle_shard,
    create_shard_tuple,
    create_torch_safetensors_shard,
    create_torch_shard,
    create_yaml_shard,
    load_from_uri,
)
from iden.testing import cloudpickle_available, safetensors_available, yaml_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if TYPE_CHECKING:
    from pathlib import Path


###################################
#     Tests for load_from_uri     #
###################################


def test_load_from_uri_file(tmp_path: Path) -> None:
    uri = tmp_path.joinpath("my_uri").as_uri()
    path = tmp_path.joinpath("my_uri.json")
    JsonSaver().save(FileShard.generate_uri_config(path), sanitize_path(uri))
    JsonSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)

    shard = load_from_uri(uri)
    assert shard.equal(FileShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})


def test_load_from_uri_json(tmp_path: Path) -> None:
    uri = tmp_path.joinpath("my_uri").as_uri()
    path = tmp_path.joinpath("my_uri.json")
    create_json_shard(data={"key1": [1, 2, 3], "key2": "abc"}, uri=uri, path=path)
    shard = load_from_uri(uri)
    assert shard.equal(JsonShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})


def test_load_from_uri_pickle(tmp_path: Path) -> None:
    uri = tmp_path.joinpath("my_uri").as_uri()
    path = tmp_path.joinpath("my_uri.pkl")
    create_pickle_shard(data={"key1": [1, 2, 3], "key2": "abc"}, uri=uri, path=path)
    shard = load_from_uri(uri)
    assert shard.equal(PickleShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})


@cloudpickle_available
def test_load_from_uri_cloudpickle(tmp_path: Path) -> None:
    uri = tmp_path.joinpath("my_uri").as_uri()
    path = tmp_path.joinpath("my_uri.pkl")
    create_cloudpickle_shard(data={"key1": [1, 2, 3], "key2": "abc"}, uri=uri, path=path)
    shard = load_from_uri(uri)
    assert shard.equal(CloudpickleShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})


@safetensors_available
@numpy_available
def test_load_from_uri_numpy_safetensors(tmp_path: Path) -> None:
    uri = tmp_path.joinpath("my_uri").as_uri()
    path = tmp_path.joinpath("my_uri.safetensors")
    create_numpy_safetensors_shard(
        data={"key1": np.ones((2, 3)), "key2": np.arange(5)}, uri=uri, path=path
    )
    shard = load_from_uri(uri)
    assert shard.equal(NumpySafetensorsShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": np.ones((2, 3)), "key2": np.arange(5)})


@safetensors_available
@torch_available
def test_load_from_uri_torch_safetensors(tmp_path: Path) -> None:
    uri = tmp_path.joinpath("my_uri").as_uri()
    path = tmp_path.joinpath("my_uri.safetensors")
    create_torch_safetensors_shard(
        data={"key1": torch.ones(2, 3), "key2": torch.arange(5)}, uri=uri, path=path
    )
    shard = load_from_uri(uri)
    assert shard.equal(TorchSafetensorsShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


@torch_available
def test_load_from_uri_torch(tmp_path: Path) -> None:
    uri = tmp_path.joinpath("my_uri").as_uri()
    path = tmp_path.joinpath("my_uri.pt")
    create_torch_shard(data={"key1": torch.ones(2, 3), "key2": torch.arange(5)}, uri=uri, path=path)
    shard = load_from_uri(uri)
    assert shard.equal(TorchShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


def test_load_from_uri_tuple(tmp_path: Path) -> None:
    shards = (
        create_json_shard([1, 2, 3], uri=tmp_path.joinpath("uri1").as_uri()),
        create_json_shard([4, 5, 6, 7], uri=tmp_path.joinpath("uri2").as_uri()),
        create_json_shard([8], uri=tmp_path.joinpath("uri3").as_uri()),
    )

    uri = tmp_path.joinpath("my_uri").as_uri()
    create_shard_tuple(shards=shards, uri=uri)

    shard = load_from_uri(uri)
    assert shard.equal(ShardTuple(uri=uri, shards=shards))
    assert objects_are_equal(
        shard.get_data(),
        (
            JsonShard.from_uri(uri=tmp_path.joinpath("uri1").as_uri()),
            JsonShard.from_uri(uri=tmp_path.joinpath("uri2").as_uri()),
            JsonShard.from_uri(uri=tmp_path.joinpath("uri3").as_uri()),
        ),
    )


@yaml_available
def test_load_from_uri_yaml(tmp_path: Path) -> None:
    uri = tmp_path.joinpath("my_uri").as_uri()
    path = tmp_path.joinpath("my_uri.yaml")
    create_yaml_shard(data={"key1": [1, 2, 3], "key2": "abc"}, uri=uri, path=path)
    shard = load_from_uri(uri)
    assert shard.equal(YamlShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})


def test_load_from_uri_missing() -> None:
    with pytest.raises(FileNotFoundError, match="uri file does not exist:"):
        load_from_uri("file:///data/my_uri")
