from __future__ import annotations

from typing import TYPE_CHECKING

from iden.shard import JsonShard, ShardTuple
from iden.shard.creator import JsonShardCreator, ShardTupleCreator

if TYPE_CHECKING:
    from pathlib import Path


#######################################
#     Tests for ShardTupleCreator     #
#######################################


def test_shard_tuple_creator_repr(tmp_path: Path) -> None:
    creator = ShardTupleCreator(
        shard=JsonShardCreator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=[1, 2, 3],
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    assert repr(creator).startswith("ShardTupleCreator(")


def test_shard_tuple_creator_str(tmp_path: Path) -> None:
    creator = ShardTupleCreator(
        shard=JsonShardCreator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=[1, 2, 3],
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    assert str(creator).startswith("ShardTupleCreator(")


def test_shard_tuple_creator_create(tmp_path: Path) -> None:
    creator = ShardTupleCreator(
        shard=JsonShardCreator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=[1, 2, 3],
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    shard = creator.create("001111")
    assert shard.equal(
        ShardTuple(
            uri=tmp_path.joinpath("001111").as_uri(),
            shards=(
                JsonShard.from_uri(
                    uri=tmp_path.joinpath("shards/uri").joinpath("000000001").as_uri()
                ),
                JsonShard.from_uri(
                    uri=tmp_path.joinpath("shards/uri").joinpath("000000002").as_uri()
                ),
                JsonShard.from_uri(
                    uri=tmp_path.joinpath("shards/uri").joinpath("000000003").as_uri()
                ),
                JsonShard.from_uri(
                    uri=tmp_path.joinpath("shards/uri").joinpath("000000004").as_uri()
                ),
            ),
        )
    )
