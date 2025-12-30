from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal

from iden.shard import CloudpickleShard, create_cloudpickle_shard
from iden.shard.loader import CloudpickleShardLoader
from iden.testing import cloudpickle_available

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.pkl")


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_cloudpickle_shard(data=[1, 2, 3], uri=uri_, path=path)
    return uri_


############################################
#     Tests for CloudpickleShardLoader     #
############################################


@cloudpickle_available
def test_cloudpickle_shard_loader_repr() -> None:
    assert repr(CloudpickleShardLoader()).startswith("CloudpickleShardLoader(")


@cloudpickle_available
def test_cloudpickle_shard_loader_str() -> None:
    assert str(CloudpickleShardLoader()).startswith("CloudpickleShardLoader(")


@cloudpickle_available
def test_cloudpickle_shard_loader_equal_true() -> None:
    assert CloudpickleShardLoader().equal(CloudpickleShardLoader())


@cloudpickle_available
def test_cloudpickle_shard_loader_equal_false_different_type() -> None:
    assert not CloudpickleShardLoader().equal(42)


@cloudpickle_available
def test_cloudpickle_shard_loader_equal_false_different_type_child() -> None:
    class Child(CloudpickleShardLoader): ...

    assert not CloudpickleShardLoader().equal(Child())


@cloudpickle_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_cloudpickle_shard_loader_equal_true_equal_nan(equal_nan: bool) -> None:
    assert CloudpickleShardLoader().equal(CloudpickleShardLoader(), equal_nan=equal_nan)


@cloudpickle_available
def test_cloudpickle_shard_loader_load(uri: str, path: Path) -> None:
    shard = CloudpickleShardLoader().load(uri)
    assert shard.equal(CloudpickleShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), [1, 2, 3])
