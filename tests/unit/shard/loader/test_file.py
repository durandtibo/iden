from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola.equality import objects_are_equal

from iden.shard import FileShard, create_json_shard
from iden.shard.loader import FileShardLoader

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


#####################################
#     Tests for FileShardLoader     #
#####################################


def test_file_shard_loader_repr() -> None:
    assert repr(FileShardLoader()).startswith("FileShardLoader(")


def test_file_shard_loader_str() -> None:
    assert str(FileShardLoader()).startswith("FileShardLoader(")


def test_file_shard_loader_equal_true() -> None:
    assert FileShardLoader().equal(FileShardLoader())


def test_file_shard_loader_equal_false_different_type() -> None:
    assert not FileShardLoader().equal(42)


def test_file_shard_loader_equal_false_different_type_child() -> None:
    class Child(FileShardLoader): ...

    assert not FileShardLoader().equal(Child())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_file_shard_loader_equal_true_equal_nan(equal_nan: bool) -> None:
    assert FileShardLoader().equal(FileShardLoader(), equal_nan=equal_nan)


def test_file_shard_loader_load(uri: str, path: Path) -> None:
    shard = FileShardLoader().load(uri)
    assert shard.equal(FileShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})
