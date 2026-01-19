from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola.equality import objects_are_equal

from iden.shard import JoblibShard, create_joblib_shard
from iden.shard.loader import JoblibShardLoader
from iden.testing import joblib_available

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.pkl")


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_joblib_shard(data=[1, 2, 3], uri=uri_, path=path)
    return uri_


#######################################
#     Tests for JoblibShardLoader     #
#######################################


@joblib_available
def test_joblib_shard_loader_repr() -> None:
    assert repr(JoblibShardLoader()).startswith("JoblibShardLoader(")


@joblib_available
def test_joblib_shard_loader_str() -> None:
    assert str(JoblibShardLoader()).startswith("JoblibShardLoader(")


@joblib_available
def test_joblib_shard_loader_equal_true() -> None:
    assert JoblibShardLoader().equal(JoblibShardLoader())


@joblib_available
def test_joblib_shard_loader_equal_false_different_type() -> None:
    assert not JoblibShardLoader().equal(42)


@joblib_available
def test_joblib_shard_loader_equal_false_different_type_child() -> None:
    class Child(JoblibShardLoader): ...

    assert not JoblibShardLoader().equal(Child())


@joblib_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_joblib_shard_loader_equal_true_equal_nan(equal_nan: bool) -> None:
    assert JoblibShardLoader().equal(JoblibShardLoader(), equal_nan=equal_nan)


@joblib_available
def test_joblib_shard_loader_load(uri: str, path: Path) -> None:
    shard = JoblibShardLoader().load(uri)
    assert shard.equal(JoblibShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), [1, 2, 3])
