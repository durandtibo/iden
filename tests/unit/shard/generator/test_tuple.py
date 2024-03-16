from __future__ import annotations

from typing import TYPE_CHECKING

from iden.data.generator import DataGenerator
from iden.shard import JsonShard, ShardTuple
from iden.shard.generator import JsonShardGenerator, ShardTupleGenerator

if TYPE_CHECKING:
    from pathlib import Path


#########################################
#     Tests for ShardTupleGenerator     #
#########################################


def test_shard_tuple_generator_repr(tmp_path: Path) -> None:
    generator = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    assert repr(generator).startswith("ShardTupleGenerator(")


def test_shard_tuple_generator_str(tmp_path: Path) -> None:
    generator = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    assert str(generator).startswith("ShardTupleGenerator(")


def test_shard_tuple_generator_generate(tmp_path: Path) -> None:
    generator = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    shard = generator.generate("001111")
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
