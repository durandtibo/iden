from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal
from objectory import OBJECT_TARGET

from iden.constants import LOADER, SHARDS
from iden.io import load_json
from iden.shard import JsonShard, ShardTuple, create_json_shard, create_shard_tuple

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from iden.shard import BaseShard


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


################################
#     Tests for ShardTuple     #
################################


def test_shard_tuple_len(uri: str, shards: Sequence[BaseShard]) -> None:
    assert len(ShardTuple(uri=uri, shards=shards)) == 3


def test_shard_tuple_repr(uri: str, shards: Sequence[BaseShard]) -> None:
    assert repr(ShardTuple(uri=uri, shards=shards)).startswith("ShardTuple(")


def test_shard_tuple_str(uri: str, shards: Sequence[BaseShard]) -> None:
    assert str(ShardTuple(uri=uri, shards=shards)).startswith("ShardTuple(")


def test_shard_tuple_equal_true(uri: str, shards: Sequence[BaseShard]) -> None:
    assert ShardTuple(uri=uri, shards=shards).equal(ShardTuple(uri=uri, shards=shards))


def test_shard_tuple_equal_false_different_uri(uri: str, shards: Sequence[BaseShard]) -> None:
    assert not ShardTuple(uri=uri, shards=shards).equal(ShardTuple(uri + "123", shards=shards))


def test_shard_tuple_equal_false_different_shards(uri: str, shards: Sequence[BaseShard]) -> None:
    assert not ShardTuple(uri=uri, shards=shards).equal(ShardTuple(uri, ()))


def test_shard_tuple_equal_false_different_type(uri: str, shards: Sequence[BaseShard]) -> None:
    assert not ShardTuple(uri=uri, shards=shards).equal([])


def test_shard_tuple_get(uri: str, shards: Sequence[BaseShard], path_shard: Path) -> None:
    sl = ShardTuple(uri=uri, shards=shards)
    assert sl.get(0).equal(JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()))
    assert sl.get(1).equal(JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()))
    assert sl.get(2).equal(JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()))


def test_shard_tuple_get_missing(uri: str, shards: Sequence[BaseShard]) -> None:
    sl = ShardTuple(uri=uri, shards=shards)
    with pytest.raises(IndexError, match="tuple index out of range"):
        sl.get(5)


def test_shard_tuple_getitem(uri: str, shards: Sequence[BaseShard], path_shard: Path) -> None:
    sl = ShardTuple(uri=uri, shards=shards)
    assert sl[0].equal(JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()))
    assert sl[1].equal(JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()))
    assert sl[2].equal(JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()))


def test_shard_tuple_get_data(uri: str, shards: Sequence[BaseShard], path_shard: Path) -> None:
    assert objects_are_equal(
        ShardTuple(uri=uri, shards=shards).get_data(),
        (
            JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        ),
    )


def test_shard_tuple_get_data_empty(
    uri: str,
) -> None:
    assert objects_are_equal(ShardTuple(uri=uri, shards=[]).get_data(), ())


def test_shard_tuple_is_sorted_by_uri_true(uri: str, shards: Sequence[BaseShard]) -> None:
    assert ShardTuple(uri=uri, shards=shards).is_sorted_by_uri()


def test_shard_tuple_is_sorted_by_uri_false(uri: str, path_shard: Path) -> None:
    assert not ShardTuple(
        uri=uri,
        shards=[
            JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        ],
    ).is_sorted_by_uri()


def test_shard_tuple_is_sorted_by_uri_empty(uri: str) -> None:
    assert ShardTuple(uri=uri, shards=[]).is_sorted_by_uri()


def test_shard_tuple_from_uri(uri: str, shards: Sequence[BaseShard], path_shard: Path) -> None:
    shard = ShardTuple.from_uri(uri)
    assert shard.equal(ShardTuple(uri=uri, shards=shards))
    assert objects_are_equal(
        shard.get_data(),
        (
            JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        ),
    )


def test_shard_tuple_generate_uri_config(shards: Sequence[BaseShard], path_shard: Path) -> None:
    assert ShardTuple.generate_uri_config(shards) == {
        SHARDS: [
            path_shard.joinpath("uri1").as_uri(),
            path_shard.joinpath("uri2").as_uri(),
            path_shard.joinpath("uri3").as_uri(),
        ],
        LOADER: {OBJECT_TARGET: "iden.shard.loader.ShardTupleLoader"},
    }


########################################
#     Tests for create_shard_tuple     #
########################################


def test_create_shard_tuple(tmp_path: Path, shards: Sequence[BaseShard], path_shard: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    shard = create_shard_tuple(shards=shards, uri=uri)

    assert uri_file.is_file()
    assert objects_are_equal(
        load_json(uri_file),
        {
            SHARDS: [
                path_shard.joinpath("uri1").as_uri(),
                path_shard.joinpath("uri2").as_uri(),
                path_shard.joinpath("uri3").as_uri(),
            ],
            LOADER: {OBJECT_TARGET: "iden.shard.loader.ShardTupleLoader"},
        },
    )
    assert shard.equal(ShardTuple(uri=uri, shards=shards))
    assert objects_are_equal(
        shard.get_data(),
        (
            JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        ),
    )
