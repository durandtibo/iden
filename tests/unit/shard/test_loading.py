from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from coola import objects_are_equal
from coola.testing import torch_available
from coola.utils import is_torch_available

from iden.shard import (
    JsonShard,
    PickleShard,
    TorchSafetensorsShard,
    TorchShard,
    YamlShard,
    create_json_shard,
    create_pickle_shard,
    create_torch_safetensors_shard,
    create_torch_shard,
    create_yaml_shard,
    load_from_uri,
)
from iden.testing import safetensors_available

if TYPE_CHECKING:
    from pathlib import Path

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


###################################
#     Tests for load_from_uri     #
###################################


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
