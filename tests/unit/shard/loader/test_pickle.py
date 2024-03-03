from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iden.shard.loader import PickleShardLoader
from iden.shard.pickle_ import PickleShard, save_uri_file
from iden.utils.io import save_pickle

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


#######################################
#     Tests for PickleShardLoader     #
#######################################


def test_pickle_shard_loader_str() -> None:
    assert str(PickleShardLoader()).startswith("PickleShardLoader(")


def test_pickle_shard_loader_load(uri: str, path: Path) -> None:
    shard = PickleShardLoader().load(uri)
    assert shard.equal(PickleShard(uri=uri, path=path))
    assert shard.get_data() == [1, 2, 3]
