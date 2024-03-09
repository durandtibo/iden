from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal

from iden.shard import JsonShard, create_json_shard
from iden.shard.collection import ShardList

if TYPE_CHECKING:
    from pathlib import Path

    from iden.shard import BaseShard


@pytest.fixture(scope="module")
def path_shard(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("shards")


@pytest.fixture(scope="module")
def shard(path_shard: Path) -> BaseShard:
    return create_json_shard([10, 11], uri=path_shard.joinpath("uri4").as_uri())


@pytest.fixture(scope="module")
def shards(path_shard: Path) -> list[BaseShard]:
    return [
        create_json_shard([1, 2, 3], uri=path_shard.joinpath("uri1").as_uri()),
        create_json_shard([4, 5, 6, 7], uri=path_shard.joinpath("uri2").as_uri()),
        create_json_shard([8], uri=path_shard.joinpath("uri3").as_uri()),
    ]


###############################
#     Tests for ShardList     #
###############################


def test_shard_list_len(shards: list[BaseShard]) -> None:
    assert len(ShardList(shards)) == 3


def test_shard_list_repr(shards: list[BaseShard]) -> None:
    assert repr(ShardList(shards)).startswith("ShardList(")


def test_shard_list_str(shards: list[BaseShard]) -> None:
    assert str(ShardList(shards)).startswith("ShardList(")


def test_shard_list_append(shards: list[BaseShard], shard: BaseShard, path_shard: Path) -> None:
    sl = ShardList(shards)
    sl.append(shard)
    assert sl.equal(
        ShardList(
            [
                JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
                JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
                JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
                JsonShard.from_uri(uri=path_shard.joinpath("uri4").as_uri()),
            ]
        )
    )


def test_shard_list_equal_true(shards: list[BaseShard]) -> None:
    assert ShardList(shards).equal(ShardList(shards))


def test_shard_list_equal_false_different_shards(shards: list[BaseShard]) -> None:
    assert not ShardList(shards).equal(ShardList())


def test_shard_list_equal_false_different_type(shards: list[BaseShard]) -> None:
    assert not ShardList(shards).equal([])


def test_shard_list_get(shards: list[BaseShard], path_shard: Path) -> None:
    sl = ShardList(shards)
    assert sl.get(0).equal(JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()))
    assert sl.get(1).equal(JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()))
    assert sl.get(2).equal(JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()))


def test_shard_list_get_missing(shards: list[BaseShard]) -> None:
    sl = ShardList(shards)
    with pytest.raises(IndexError, match="list index out of range"):
        sl.get(5)


def test_shard_list_getitem(shards: list[BaseShard], path_shard: Path) -> None:
    sl = ShardList(shards)
    assert sl[0].equal(JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()))
    assert sl[1].equal(JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()))
    assert sl[2].equal(JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()))


def test_shard_list_get_shards(shards: list[BaseShard], path_shard: Path) -> None:
    assert objects_are_equal(
        ShardList(shards).get_shards(),
        [
            JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        ],
    )


def test_shard_list_get_shards_empty() -> None:
    assert objects_are_equal(ShardList().get_shards(), [])


def test_shard_list_pop(shards: list[BaseShard], path_shard: Path) -> None:
    sl = ShardList(shards)
    shard = sl.pop(1)
    assert shard.equal(JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()))
    assert sl.equal(
        ShardList(
            [
                JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
                JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
            ]
        )
    )


def test_shard_list_pop_empty() -> None:
    sl = ShardList()
    with pytest.raises(IndexError, match="pop from empty list"):
        sl.pop(0)


def test_shard_list_pop_missing_index(shards: list[BaseShard]) -> None:
    sl = ShardList(shards)
    with pytest.raises(IndexError, match="pop index out of range"):
        sl.pop(5)
