from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal
from objectory import OBJECT_TARGET

from iden.constants import KWARGS, LOADER
from iden.io import load_json
from iden.shard import JsonShard, create_json_shard

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.json")


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_json_shard(data={"key1": [1, 2, 3], "key2": "abc"}, uri=uri_, path=path)
    return uri_


###############################
#     Tests for JsonShard     #
###############################


def test_json_shard_str(uri: str, path: Path) -> None:
    assert str(JsonShard(uri=uri, path=path)).startswith("JsonShard(")


def test_json_shard_path(uri: str, path: Path) -> None:
    assert JsonShard(uri=uri, path=path).path == path


def test_json_shard_equal_true(uri: str, path: Path) -> None:
    assert JsonShard(uri=uri, path=path).equal(JsonShard(uri=uri, path=path))


def test_json_shard_equal_false_different_uri(uri: str, path: Path) -> None:
    assert not JsonShard(uri=uri, path=path).equal(JsonShard(uri="", path=path))


def test_json_shard_equal_false_different_path(uri: str, path: Path, tmp_path: Path) -> None:
    assert not JsonShard(uri=uri, path=path).equal(JsonShard(uri=uri, path=tmp_path))


def test_json_shard_equal_false_different_type(uri: str, path: Path) -> None:
    assert not JsonShard(uri=uri, path=path).equal(42)


def test_json_shard_get_data(uri: str, path: Path) -> None:
    assert objects_are_equal(
        JsonShard(uri=uri, path=path).get_data(), {"key1": [1, 2, 3], "key2": "abc"}
    )


def test_json_shard_get_data_multiple_calls(uri: str, path: Path) -> None:
    shard = JsonShard(uri=uri, path=path)
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})
    shard.get_data()["key1"].append(4)
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3, 4], "key2": "abc"})


def test_json_shard_get_uri(uri: str, path: Path) -> None:
    assert JsonShard(uri=uri, path=path).get_uri() == uri


def test_json_shard_from_uri(uri: str, path: Path) -> None:
    shard = JsonShard.from_uri(uri)
    assert shard.equal(JsonShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})


def test_json_shard_generate_uri_config(path: Path) -> None:
    assert JsonShard.generate_uri_config(path) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.JsonShardLoader"},
    }


#######################################
#     Tests for create_json_shard     #
#######################################


def test_create_json_shard(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("my_uri.json")
    shard = create_json_shard(data={"key1": [1, 2, 3], "key2": "abc"}, uri=uri)

    assert uri_file.is_file()
    assert load_json(uri_file) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.JsonShardLoader"},
    }
    assert shard.equal(JsonShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})


def test_create_json_shard_with_data(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("data.json")
    shard = create_json_shard(data={"key1": [1, 2, 3], "key2": "abc"}, uri=uri, path=path)

    assert load_json(uri_file) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.JsonShardLoader"},
    }
    assert shard.equal(JsonShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})
