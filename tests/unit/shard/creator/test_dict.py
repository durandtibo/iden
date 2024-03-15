from __future__ import annotations

from typing import TYPE_CHECKING

from iden.shard import JsonShard, ShardDict
from iden.shard.creator import JsonShardCreator, ShardDictCreator

if TYPE_CHECKING:
    from pathlib import Path


######################################
#     Tests for ShardDictCreator     #
######################################


def test_shard_dict_creator_repr(tmp_path: Path) -> None:
    creator = ShardDictCreator(
        shards={
            "train": JsonShardCreator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=[1, 2, 3],
            ),
            "val": JsonShardCreator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=[4, 5, 6],
            ),
        },
        path_uri=tmp_path,
    )
    assert repr(creator).startswith("ShardDictCreator(")


def test_shard_dict_creator_str(tmp_path: Path) -> None:
    creator = ShardDictCreator(
        shards={
            "train": JsonShardCreator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=[1, 2, 3],
            ),
            "val": JsonShardCreator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=[4, 5, 6],
            ),
        },
        path_uri=tmp_path,
    )
    assert str(creator).startswith("ShardDictCreator(")


def test_shard_dict_creator_create(tmp_path: Path) -> None:
    creator = ShardDictCreator(
        shards={
            "train": JsonShardCreator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=[1, 2, 3],
            ),
            "val": JsonShardCreator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=[4, 5, 6],
            ),
        },
        path_uri=tmp_path,
    )
    shard = creator.create("001111")
    assert shard.equal(
        ShardDict(
            uri=tmp_path.joinpath("001111").as_uri(),
            shards={
                "train": JsonShard.from_uri(uri=tmp_path.joinpath("shards/uri/train").as_uri()),
                "val": JsonShard.from_uri(uri=tmp_path.joinpath("shards/uri/val").as_uri()),
            },
        ),
    )
