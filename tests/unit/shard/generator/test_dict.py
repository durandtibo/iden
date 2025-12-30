from __future__ import annotations

from typing import TYPE_CHECKING

from iden.data.generator import DataGenerator
from iden.shard import JsonShard, ShardDict
from iden.shard.generator import JsonShardGenerator, ShardDictGenerator

if TYPE_CHECKING:
    from pathlib import Path


########################################
#     Tests for ShardDictGenerator     #
########################################


def test_shard_dict_generator_repr(tmp_path: Path) -> None:
    generator = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
            "val": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([4, 5, 6]),
            ),
        },
        path_uri=tmp_path,
    )
    assert repr(generator).startswith("ShardDictGenerator(")


def test_shard_dict_generator_str(tmp_path: Path) -> None:
    generator = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
            "val": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([4, 5, 6]),
            ),
        },
        path_uri=tmp_path,
    )
    assert str(generator).startswith("ShardDictGenerator(")


def test_shard_dict_generator_equal_true(tmp_path: Path) -> None:
    generator1 = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
        },
        path_uri=tmp_path,
    )
    generator2 = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
        },
        path_uri=tmp_path,
    )
    assert generator1.equal(generator2)


def test_shard_dict_generator_equal_false_different_shards(tmp_path: Path) -> None:
    generator1 = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
        },
        path_uri=tmp_path,
    )
    generator2 = ShardDictGenerator(
        shards={
            "val": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
        },
        path_uri=tmp_path,
    )
    assert not generator1.equal(generator2)


def test_shard_dict_generator_equal_false_different_path_uri(tmp_path: Path) -> None:
    generator1 = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
        },
        path_uri=tmp_path,
    )
    generator2 = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
        },
        path_uri=tmp_path.joinpath("other"),
    )
    assert not generator1.equal(generator2)


def test_shard_dict_generator_equal_false_different_type(tmp_path: Path) -> None:
    generator = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
        },
        path_uri=tmp_path,
    )
    assert not generator.equal(42)


def test_shard_dict_generator_equal_false_different_type_child(tmp_path: Path) -> None:
    class Child(ShardDictGenerator): ...

    generator1 = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
        },
        path_uri=tmp_path,
    )
    generator2 = Child(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
        },
        path_uri=tmp_path,
    )
    assert not generator1.equal(generator2)


def test_shard_dict_generator_equal_true_equal_nan(tmp_path: Path) -> None:
    generator1 = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3, float("nan")]),
            ),
        },
        path_uri=tmp_path,
    )
    generator2 = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3, float("nan")]),
            ),
        },
        path_uri=tmp_path,
    )
    assert generator1.equal(generator2, equal_nan=True)


def test_shard_dict_generator_equal_false_equal_nan(tmp_path: Path) -> None:
    generator1 = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3, float("nan")]),
            ),
        },
        path_uri=tmp_path,
    )
    generator2 = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3, float("nan")]),
            ),
        },
        path_uri=tmp_path,
    )
    assert not generator1.equal(generator2)


def test_shard_dict_generator_generate(tmp_path: Path) -> None:
    generator = ShardDictGenerator(
        shards={
            "train": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([1, 2, 3]),
            ),
            "val": JsonShardGenerator(
                path_shard=tmp_path.joinpath("shards/data"),
                path_uri=tmp_path.joinpath("shards/uri"),
                data=DataGenerator([4, 5, 6]),
            ),
        },
        path_uri=tmp_path,
    )
    shard = generator.generate("001111")
    assert shard.equal(
        ShardDict(
            uri=tmp_path.joinpath("001111").as_uri(),
            shards={
                "train": JsonShard.from_uri(uri=tmp_path.joinpath("shards/uri/train").as_uri()),
                "val": JsonShard.from_uri(uri=tmp_path.joinpath("shards/uri/val").as_uri()),
            },
        ),
    )
