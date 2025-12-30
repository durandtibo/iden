from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal

from iden.shard import PickleShard, create_pickle_shard
from iden.shard.loader import PickleShardLoader

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.pkl")


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_pickle_shard(data=[1, 2, 3], uri=uri_, path=path)
    return uri_


#######################################
#     Tests for PickleShardLoader     #
#######################################


def test_pickle_shard_loader_repr() -> None:
    assert repr(PickleShardLoader()).startswith("PickleShardLoader(")


def test_pickle_shard_loader_str() -> None:
    assert str(PickleShardLoader()).startswith("PickleShardLoader(")


def test_pickle_shard_loader_equal_true() -> None:
    assert PickleShardLoader().equal(PickleShardLoader())


def test_pickle_shard_loader_equal_false_different_type() -> None:
    assert not PickleShardLoader().equal(42)


def test_pickle_shard_loader_equal_false_different_type_child() -> None:
    class Child(PickleShardLoader): ...

    assert not PickleShardLoader().equal(Child())


def test_pickle_shard_loader_load(uri: str, path: Path) -> None:
    shard = PickleShardLoader().load(uri)
    assert shard.equal(PickleShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), [1, 2, 3])
