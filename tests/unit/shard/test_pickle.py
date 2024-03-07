from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from objectory import OBJECT_TARGET

from iden.constants import KWARGS, LOADER
from iden.io import load_json
from iden.shard import PickleShard
from iden.shard.pickle import create_pickle_shard

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


def test_pickle_shard_generate_uri_config(path: Path) -> None:
    assert PickleShard.generate_uri_config(path) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.PickleShardLoader"},
    }


#########################################
#     Tests for create_pickle_shard     #
#########################################


def test_create_pickle_shard(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("my_uri.pkl")
    shard = create_pickle_shard(data=[1, 2, 3], uri=uri)

    assert uri_file.is_file()
    assert load_json(uri_file) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.PickleShardLoader"},
    }
    assert shard.equal(PickleShard(uri=uri, path=path))
    assert shard.get_data() == [1, 2, 3]


def test_create_pickle_shard_with_data(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("data.pkl")
    shard = create_pickle_shard(data=[1, 2, 3], uri=uri, path=path)

    assert load_json(uri_file) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.PickleShardLoader"},
    }
    assert shard.equal(PickleShard(uri=uri, path=path))
    assert shard.get_data() == [1, 2, 3]
