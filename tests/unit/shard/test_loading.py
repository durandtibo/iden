from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iden.shard import PickleShard, create_pickle_shard, load_from_uri

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


###################################
#     Tests for load_from_uri     #
###################################


def test_load_from_uri(uri: str, path: Path) -> None:
    shard = load_from_uri(uri)
    assert shard.equal(PickleShard(uri=uri, path=path))
    assert shard.get_data() == [1, 2, 3]


def test_load_from_uri_missing() -> None:
    with pytest.raises(FileNotFoundError, match="uri file does not exist:"):
        load_from_uri("file:///data/my_uri")
