from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal
from coola.utils.path import sanitize_path
from objectory import OBJECT_TARGET

from iden.constants import KWARGS, LOADER
from iden.io import save_json
from iden.shard import FileShard, create_json_shard

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path_ = tmp_path_factory.mktemp("tmp").joinpath("data.json")
    save_json({"key1": [1, 2, 3], "key2": "abc"}, path_)
    return path_


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    config = {KWARGS: {"path": sanitize_path(path).as_posix()}, LOADER: ""}
    save_json(config, sanitize_path(uri_))
    return uri_


###############################
#     Tests for FileShard     #
###############################


def test_file_shard_str(uri: str, path: Path) -> None:
    assert str(FileShard(uri=uri, path=path)).startswith("FileShard(")


def test_file_shard_path(uri: str, path: Path) -> None:
    assert FileShard(uri=uri, path=path).path == path


def test_file_shard_clear_not_initialized(uri: str, path: Path) -> None:
    shard = FileShard(uri=uri, path=path)
    shard.clear()
    assert not shard.is_cached()
    assert shard._data is None


def test_file_shard_clear_initialized(uri: str, path: Path) -> None:
    shard = FileShard(uri=uri, path=path)
    assert objects_are_equal(shard.get_data(cache=True), {"key1": [1, 2, 3], "key2": "abc"})
    assert shard.is_cached()
    assert objects_are_equal(shard._data, {"key1": [1, 2, 3], "key2": "abc"})

    shard.clear()
    assert not shard.is_cached()
    assert shard._data is None


def test_file_shard_equal_true(uri: str, path: Path) -> None:
    assert FileShard(uri=uri, path=path).equal(FileShard(uri=uri, path=path))


def test_file_shard_equal_false_different_uri(uri: str, path: Path) -> None:
    assert not FileShard(uri=uri, path=path).equal(FileShard(uri="", path=path))


def test_file_shard_equal_false_different_path(uri: str, path: Path, tmp_path: Path) -> None:
    assert not FileShard(uri=uri, path=path).equal(FileShard(uri=uri, path=tmp_path))


def test_file_shard_equal_false_different_type(uri: str, path: Path) -> None:
    assert not FileShard(uri=uri, path=path).equal(42)


@pytest.mark.parametrize("equal_nan", [True, False])
def test_file_shard_equal_nan(tmp_path: Path, equal_nan: bool) -> None:
    shard = create_json_shard(
        data={"key1": [1, 2, float("nan")], "key2": "abc"}, uri=tmp_path.joinpath("uri").as_uri()
    )
    assert FileShard.from_uri(uri=shard.get_uri()).equal(
        FileShard.from_uri(uri=shard.get_uri()), equal_nan=equal_nan
    )


def test_file_shard_get_data(uri: str, path: Path) -> None:
    assert FileShard(uri=uri, path=path).get_data() == {"key1": [1, 2, 3], "key2": "abc"}


def test_file_shard_get_data_cache_false_not_cached(uri: str, path: Path) -> None:
    shard = FileShard(uri=uri, path=path)
    assert not shard.is_cached()
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})
    assert not shard.is_cached()


def test_file_shard_get_data_cache_false_cached(uri: str, path: Path) -> None:
    shard = FileShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})
    assert shard.is_cached()


def test_file_shard_get_data_cache_true_not_cached(uri: str, path: Path) -> None:
    shard = FileShard(uri=uri, path=path)
    assert not shard.is_cached()
    assert objects_are_equal(shard.get_data(cache=True), {"key1": [1, 2, 3], "key2": "abc"})
    assert shard.is_cached()


def test_file_shard_get_data_cache_true_cached(uri: str, path: Path) -> None:
    shard = FileShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()
    assert objects_are_equal(shard.get_data(cache=True), {"key1": [1, 2, 3], "key2": "abc"})
    assert shard.is_cached()


def test_file_shard_get_data_multiple_calls_cache(uri: str, path: Path) -> None:
    shard = FileShard(uri=uri, path=path)
    assert shard.get_data(cache=True) == {"key1": [1, 2, 3], "key2": "abc"}
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})
    shard.get_data()["key1"].append(4)
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3, 4], "key2": "abc"})


def test_file_shard_get_uri(uri: str, path: Path) -> None:
    assert FileShard(uri=uri, path=path).get_uri() == uri


def test_file_shard_is_cached_false(uri: str, path: Path) -> None:
    shard = FileShard(uri=uri, path=path)
    assert not shard.is_cached()


def test_file_shard_is_cached_true(uri: str, path: Path) -> None:
    shard = FileShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()


def test_file_shard_from_uri(uri: str, path: Path) -> None:
    shard = FileShard.from_uri(uri)
    assert shard.equal(FileShard(uri=uri, path=path))
    assert shard.get_data() == {"key1": [1, 2, 3], "key2": "abc"}


def test_json_shard_generate_uri_config(path: Path) -> None:
    assert FileShard.generate_uri_config(path) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.FileShardLoader"},
    }
