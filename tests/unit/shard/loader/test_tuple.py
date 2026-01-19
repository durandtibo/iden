from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola.equality import objects_are_equal

from iden.shard import (
    BaseShard,
    JsonShard,
    ShardTuple,
    create_json_shard,
    create_shard_tuple,
)
from iden.shard.loader import ShardTupleLoader

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@pytest.fixture(scope="module")
def path_shard(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("shards")


@pytest.fixture(scope="module")
def shards(path_shard: Path) -> tuple[BaseShard, ...]:
    return (
        create_json_shard([1, 2, 3], uri=path_shard.joinpath("uri1").as_uri()),
        create_json_shard([4, 5, 6, 7], uri=path_shard.joinpath("uri2").as_uri()),
        create_json_shard([8], uri=path_shard.joinpath("uri3").as_uri()),
    )


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, shards: Sequence[BaseShard]) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_shard_tuple(shards=shards, uri=uri_)
    return uri_


######################################
#     Tests for ShardTupleLoader     #
######################################


def test_shard_tuple_loader_repr() -> None:
    assert repr(ShardTupleLoader()).startswith("ShardTupleLoader(")


def test_shard_tuple_loader_str() -> None:
    assert str(ShardTupleLoader()).startswith("ShardTupleLoader(")


def test_shard_tuple_loader_equal_true() -> None:
    assert ShardTupleLoader().equal(ShardTupleLoader())


def test_shard_tuple_loader_equal_false_different_type() -> None:
    assert not ShardTupleLoader().equal(42)


def test_shard_tuple_loader_equal_false_different_type_child() -> None:
    class Child(ShardTupleLoader): ...

    assert not ShardTupleLoader().equal(Child())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_shard_tuple_loader_equal_true_equal_nan(equal_nan: bool) -> None:
    assert ShardTupleLoader().equal(ShardTupleLoader(), equal_nan=equal_nan)


def test_shard_tuple_loader_load(uri: str, shards: Sequence[BaseShard], path_shard: Path) -> None:
    shard = ShardTupleLoader().load(uri)
    assert shard.equal(ShardTuple(uri=uri, shards=shards))
    assert objects_are_equal(
        shard.get_data(),
        (
            JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        ),
    )
