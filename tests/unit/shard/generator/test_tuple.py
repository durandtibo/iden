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


def test_shard_tuple_generator_equal_true(tmp_path: Path) -> None:
    generator1 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    generator2 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    assert generator1.equal(generator2)


def test_shard_tuple_generator_equal_false_different_shard(tmp_path: Path) -> None:
    generator1 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    generator2 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    assert not generator1.equal(generator2)


def test_shard_tuple_generator_equal_false_different_num_shards(tmp_path: Path) -> None:
    generator1 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    generator2 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=1,
        path_uri=tmp_path,
    )
    assert not generator1.equal(generator2)


def test_shard_tuple_generator_equal_false_different_path_uri(tmp_path: Path) -> None:
    generator1 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    generator2 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path.joinpath("other"),
    )
    assert not generator1.equal(generator2)


def test_shard_tuple_generator_equal_false_different_type(tmp_path: Path) -> None:
    generator = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    assert not generator.equal(42)


def test_shard_tuple_generator_equal_false_different_type_child(tmp_path: Path) -> None:
    class Child(ShardTupleGenerator): ...

    generator1 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    generator2 = Child(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    assert not generator1.equal(generator2)


def test_shard_tuple_generator_equal_true_equal_nan(tmp_path: Path) -> None:
    generator1 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3, float("nan")]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    generator2 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3, float("nan")]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    assert generator1.equal(generator2, equal_nan=True)


def test_shard_tuple_generator_equal_false_equal_nan(tmp_path: Path) -> None:
    generator1 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3, float("nan")]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    generator2 = ShardTupleGenerator(
        shard=JsonShardGenerator(
            path_shard=tmp_path.joinpath("shards/data"),
            path_uri=tmp_path.joinpath("shards/uri"),
            data=DataGenerator([1, 2, 3, float("nan")]),
        ),
        num_shards=4,
        path_uri=tmp_path,
    )
    assert not generator1.equal(generator2)


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
