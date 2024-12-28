from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal
from objectory import OBJECT_TARGET

from iden.constants import KWARGS, LOADER
from iden.io import load_json
from iden.shard import JoblibShard, create_joblib_shard
from iden.testing import joblib_available

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.joblib")


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_joblib_shard(data={"key1": [1, 2, 3], "key2": "abc"}, uri=uri_, path=path)
    return uri_


#################################
#     Tests for JoblibShard     #
#################################


@joblib_available
def test_joblib_shard_str(uri: str, path: Path) -> None:
    assert str(JoblibShard(uri=uri, path=path)).startswith("JoblibShard(")


@joblib_available
def test_joblib_shard_path(uri: str, path: Path) -> None:
    assert JoblibShard(uri=uri, path=path).path == path


@joblib_available
def test_joblib_shard_clear_not_initialized(uri: str, path: Path) -> None:
    shard = JoblibShard(uri=uri, path=path)
    shard.clear()
    assert not shard.is_cached()


@joblib_available
def test_joblib_shard_clear_is_cached(uri: str, path: Path) -> None:
    shard = JoblibShard(uri=uri, path=path)
    assert objects_are_equal(shard.get_data(cache=True), {"key1": [1, 2, 3], "key2": "abc"})
    assert shard.is_cached()
    shard.clear()
    assert not shard.is_cached()


@joblib_available
def test_joblib_shard_equal_true(uri: str, path: Path) -> None:
    assert JoblibShard(uri=uri, path=path).equal(JoblibShard(uri=uri, path=path))


@joblib_available
def test_joblib_shard_equal_false_different_uri(uri: str, path: Path) -> None:
    assert not JoblibShard(uri=uri, path=path).equal(JoblibShard(uri="", path=path))


@joblib_available
def test_joblib_shard_equal_false_different_path(uri: str, path: Path, tmp_path: Path) -> None:
    assert not JoblibShard(uri=uri, path=path).equal(JoblibShard(uri=uri, path=tmp_path))


@joblib_available
def test_joblib_shard_equal_false_different_type(uri: str, path: Path) -> None:
    assert not JoblibShard(uri=uri, path=path).equal(42)


@joblib_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_joblib_shard_equal_nan(tmp_path: Path, equal_nan: bool) -> None:
    shard = create_joblib_shard(
        data={"key1": [1, 2, float("nan")], "key2": "abc"}, uri=tmp_path.joinpath("uri").as_uri()
    )
    assert shard.equal(JoblibShard.from_uri(uri=shard.get_uri()), equal_nan=equal_nan)


@joblib_available
def test_joblib_shard_get_data(uri: str, path: Path) -> None:
    assert objects_are_equal(
        JoblibShard(uri=uri, path=path).get_data(), {"key1": [1, 2, 3], "key2": "abc"}
    )


@joblib_available
def test_joblib_shard_get_data_cache_false_not_cached(uri: str, path: Path) -> None:
    shard = JoblibShard(uri=uri, path=path)
    assert not shard.is_cached()
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})
    assert not shard.is_cached()


@joblib_available
def test_joblib_shard_get_data_cache_false_cached(uri: str, path: Path) -> None:
    shard = JoblibShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})
    assert shard.is_cached()


@joblib_available
def test_joblib_shard_get_data_cache_true_not_cached(uri: str, path: Path) -> None:
    shard = JoblibShard(uri=uri, path=path)
    assert not shard.is_cached()
    assert objects_are_equal(shard.get_data(cache=True), {"key1": [1, 2, 3], "key2": "abc"})
    assert shard.is_cached()


@joblib_available
def test_joblib_shard_get_data_cache_true_cached(uri: str, path: Path) -> None:
    shard = JoblibShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()
    assert objects_are_equal(shard.get_data(cache=True), {"key1": [1, 2, 3], "key2": "abc"})
    assert shard.is_cached()


@joblib_available
def test_joblib_shard_get_data_multiple_calls(uri: str, path: Path) -> None:
    shard = JoblibShard(uri=uri, path=path)
    assert objects_are_equal(shard.get_data(cache=True), {"key1": [1, 2, 3], "key2": "abc"})
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})
    shard.get_data()["key1"].append(4)
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3, 4], "key2": "abc"})


@joblib_available
def test_joblib_shard_get_uri(uri: str, path: Path) -> None:
    assert JoblibShard(uri=uri, path=path).get_uri() == uri


@joblib_available
def test_joblib_shard_is_cached_false(uri: str, path: Path) -> None:
    shard = JoblibShard(uri=uri, path=path)
    assert not shard.is_cached()


@joblib_available
def test_joblib_shard_is_cached_true(uri: str, path: Path) -> None:
    shard = JoblibShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()


@joblib_available
def test_joblib_shard_from_uri(uri: str, path: Path) -> None:
    shard = JoblibShard.from_uri(uri)
    assert shard.equal(JoblibShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})


@joblib_available
def test_joblib_shard_generate_uri_config(path: Path) -> None:
    assert JoblibShard.generate_uri_config(path) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.JoblibShardLoader"},
    }


#########################################
#     Tests for create_joblib_shard     #
#########################################


@joblib_available
def test_create_joblib_shard(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("my_uri.joblib")
    shard = create_joblib_shard(data={"key1": [1, 2, 3], "key2": "abc"}, uri=uri)

    assert uri_file.is_file()
    assert objects_are_equal(
        load_json(uri_file),
        {
            KWARGS: {"path": path.as_posix()},
            LOADER: {OBJECT_TARGET: "iden.shard.loader.JoblibShardLoader"},
        },
        show_difference=True,
    )
    assert shard.equal(JoblibShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})


@joblib_available
def test_create_joblib_shard_with_data(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("data.joblib")
    shard = create_joblib_shard(data={"key1": [1, 2, 3], "key2": "abc"}, uri=uri, path=path)

    assert load_json(uri_file) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.JoblibShardLoader"},
    }
    assert shard.equal(JoblibShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})
