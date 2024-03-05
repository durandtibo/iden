from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from objectory import OBJECT_TARGET

from iden.constants import KWARGS, LOADER
from iden.io import load_json, save_pickle
from iden.shard import PickleShard
from iden.shard.pickle import create_pickle_shard, save_uri_file

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path_ = tmp_path_factory.mktemp("tmp").joinpath("data.pkl")
    save_pickle([1, 2, 3], path_)
    return path_


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    save_uri_file(uri=uri_, path=path)
    return uri_


#################################
#     Tests for PickleShard     #
#################################


def test_pickle_shard_str(uri: str, path: Path) -> None:
    assert str(PickleShard(uri=uri, path=path)).startswith("PickleShard(")


def test_pickle_shard_path(uri: str, path: Path) -> None:
    assert PickleShard(uri=uri, path=path).path == path


def test_pickle_shard_equal_true(uri: str, path: Path) -> None:
    assert PickleShard(uri=uri, path=path).equal(PickleShard(uri=uri, path=path))


def test_pickle_shard_equal_false_different_uri(uri: str, path: Path) -> None:
    assert not PickleShard(uri=uri, path=path).equal(PickleShard(uri="", path=path))


def test_pickle_shard_equal_false_different_path(uri: str, path: Path, tmp_path: Path) -> None:
    assert not PickleShard(uri=uri, path=path).equal(PickleShard(uri=uri, path=tmp_path))


def test_pickle_shard_equal_false_different_type(uri: str, path: Path) -> None:
    assert not PickleShard(uri=uri, path=path).equal(42)


def test_pickle_shard_get_data(uri: str, path: Path) -> None:
    assert PickleShard(uri=uri, path=path).get_data() == [1, 2, 3]


def test_pickle_shard_get_data_multiple_calls(uri: str, path: Path) -> None:
    shard = PickleShard(uri=uri, path=path)
    assert shard.get_data() == [1, 2, 3]
    assert shard.get_data() == [1, 2, 3]
    shard.get_data().append(4)
    assert shard.get_data() == [1, 2, 3, 4]


def test_pickle_shard_get_uri(uri: str, path: Path) -> None:
    assert PickleShard(uri=uri, path=path).get_uri() == uri


def test_pickle_shard_from_uri(uri: str, path: Path) -> None:
    shard = PickleShard.from_uri(uri)
    assert shard.equal(PickleShard(uri=uri, path=path))
    assert shard.get_data() == [1, 2, 3]


#########################################
#     Tests for create_pickle_shard     #
#########################################


def test_create_pickle_shard(tmp_path: Path) -> None:
    uri = tmp_path.joinpath("my_uri").as_uri()
    shard = create_pickle_shard([1, 2, 3], uri)
    assert shard.equal(PickleShard(uri=uri, path=tmp_path.joinpath("my_uri.pkl")))
    assert shard.get_data() == [1, 2, 3]


###################################
#     Tests for save_uri_file     #
###################################


def test_save_uri_file(tmp_path: Path) -> None:
    uri = tmp_path.joinpath("uri")
    path = tmp_path.joinpath("data.pkl")
    save_uri_file(uri=uri.as_uri(), path=path)
    assert uri.is_file()
    assert load_json(uri) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.PickleShardLoader"},
    }
