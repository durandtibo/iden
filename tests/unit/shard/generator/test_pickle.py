from __future__ import annotations

from typing import TYPE_CHECKING

from coola import objects_are_equal

from iden.data.generator import DataGenerator
from iden.shard import PickleShard
from iden.shard.generator import PickleShardGenerator

if TYPE_CHECKING:
    from pathlib import Path

##########################################
#     Tests for PickleShardGenerator     #
##########################################


def test_pickle_shard_generator_repr(tmp_path: Path) -> None:
    assert repr(
        PickleShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("PickleShardGenerator(")


def test_pickle_shard_generator_str(tmp_path: Path) -> None:
    assert str(
        PickleShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("PickleShardGenerator(")


def test_pickle_shard_generator_equal_true(tmp_path: Path) -> None:
    generator1 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert generator1.equal(generator2)


def test_pickle_shard_generator_equal_false_different_data(tmp_path: Path) -> None:
    generator1 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = PickleShardGenerator(
        data=DataGenerator([]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert not generator1.equal(generator2)


def test_pickle_shard_generator_equal_false_different_path_uri(tmp_path: Path) -> None:
    generator1 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("other/uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert not generator1.equal(generator2)


def test_pickle_shard_generator_equal_false_different_path_shard(tmp_path: Path) -> None:
    generator1 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("other/shard"),
    )
    assert not generator1.equal(generator2)


def test_pickle_shard_generator_equal_false_different_type(tmp_path: Path) -> None:
    generator = PickleShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert not generator.equal(42)


def test_pickle_shard_generator_equal_false_different_type_child(tmp_path: Path) -> None:
    class Child(PickleShardGenerator): ...

    generator1 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = Child(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert not generator1.equal(generator2)


def test_pickle_shard_generator_equal_true_equal_nan(tmp_path: Path) -> None:
    generator1 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3, float("nan")]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3, float("nan")]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert generator1.equal(generator2, equal_nan=True)


def test_pickle_shard_generator_equal_false_equal_nan(tmp_path: Path) -> None:
    generator1 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3, float("nan")]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = PickleShardGenerator(
        data=DataGenerator([1, 2, 3, float("nan")]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert not generator1.equal(generator2)


def test_pickle_shard_generator_generate(tmp_path: Path) -> None:
    generator = PickleShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    shard = generator.generate("000001")
    assert shard.equal(
        PickleShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.pkl"),
        )
    )
    assert objects_are_equal(shard.get_data(), [1, 2, 3])
