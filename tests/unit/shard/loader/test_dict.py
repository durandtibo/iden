from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola.equality import objects_are_equal

from iden.shard import (
    BaseShard,
    JsonShard,
    ShardDict,
    create_json_shard,
    create_shard_dict,
)
from iden.shard.loader import ShardDictLoader

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_shard(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("shards")


@pytest.fixture(scope="module")
def shards(path_shard: Path) -> dict[str, BaseShard]:
    return {
        "001": create_json_shard([1, 2, 3], uri=path_shard.joinpath("uri1").as_uri()),
        "002": create_json_shard([4, 5, 6, 7], uri=path_shard.joinpath("uri2").as_uri()),
        "003": create_json_shard([8], uri=path_shard.joinpath("uri3").as_uri()),
    }


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, shards: dict[str, BaseShard]) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_shard_dict(shards=shards, uri=uri_)
    return uri_


#####################################
#     Tests for ShardDictLoader     #
#####################################


def test_shard_dict_loader_repr() -> None:
    assert repr(ShardDictLoader()).startswith("ShardDictLoader(")


def test_shard_dict_loader_str() -> None:
    assert str(ShardDictLoader()).startswith("ShardDictLoader(")


def test_shard_dict_loader_equal_true() -> None:
    assert ShardDictLoader().equal(ShardDictLoader())


def test_shard_dict_loader_equal_false_different_type() -> None:
    assert not ShardDictLoader().equal(42)


def test_shard_dict_loader_equal_false_different_type_child() -> None:
    class Child(ShardDictLoader): ...

    assert not ShardDictLoader().equal(Child())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_shard_dict_loader_equal_true_equal_nan(equal_nan: bool) -> None:
    assert ShardDictLoader().equal(ShardDictLoader(), equal_nan=equal_nan)


def test_shard_dict_loader_load(uri: str, shards: dict[str, BaseShard], path_shard: Path) -> None:
    shard = ShardDictLoader().load(uri)
    assert shard.equal(ShardDict(uri=uri, shards=shards))
    assert objects_are_equal(
        shard.get_data(),
        {
            "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            "002": JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            "003": JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        },
    )
