from __future__ import annotations

from typing import TYPE_CHECKING

from coola.equality import objects_are_equal

from iden.data.generator import DataGenerator
from iden.dataset.generator import VanillaDatasetGenerator
from iden.shard.generator import (
    JsonShardGenerator,
    ShardDictGenerator,
    ShardTupleGenerator,
)

if TYPE_CHECKING:
    from pathlib import Path

#######################################
#     Tests for objects_are_equal     #
#######################################


def test_objects_are_equal_true(tmp_path: Path) -> None:
    generator1 = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(path_uri=tmp_path.joinpath("uri/shards"), shards={}),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("uri/assets")),
    )
    generator2 = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(path_uri=tmp_path.joinpath("uri/shards"), shards={}),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("uri/assets")),
    )
    assert objects_are_equal(generator1, generator2)


def test_objects_are_equal_false(tmp_path: Path) -> None:
    generator1 = VanillaDatasetGenerator(
        path_uri=tmp_path.joinpath("one"),
        shards=ShardDictGenerator(path_uri=tmp_path.joinpath("uri/shards"), shards={}),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("uri/assets")),
    )
    generator2 = VanillaDatasetGenerator(
        path_uri=tmp_path.joinpath("two"),
        shards=ShardDictGenerator(path_uri=tmp_path.joinpath("uri/shards"), shards={}),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("uri/assets")),
    )
    assert not objects_are_equal(generator1, generator2)


def test_objects_are_equal_true_equal_nan(tmp_path: Path) -> None:
    generator1 = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(
            path_uri=tmp_path.joinpath("uri/shards"),
            shards={
                "train": ShardTupleGenerator(
                    shard=JsonShardGenerator(
                        path_shard=tmp_path.joinpath("data/shards/train/shards"),
                        path_uri=tmp_path.joinpath("uri/shards/train/shards"),
                        data=DataGenerator([1, 2, 3, float("nan")]),
                    ),
                    num_shards=2,
                    path_uri=tmp_path.joinpath("uri/shards/train"),
                ),
            },
        ),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("uri/assets")),
    )
    generator2 = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(
            path_uri=tmp_path.joinpath("uri/shards"),
            shards={
                "train": ShardTupleGenerator(
                    shard=JsonShardGenerator(
                        path_shard=tmp_path.joinpath("data/shards/train/shards"),
                        path_uri=tmp_path.joinpath("uri/shards/train/shards"),
                        data=DataGenerator([1, 2, 3, float("nan")]),
                    ),
                    num_shards=2,
                    path_uri=tmp_path.joinpath("uri/shards/train"),
                ),
            },
        ),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("uri/assets")),
    )
    assert objects_are_equal(generator1, generator2, equal_nan=True)


def test_objects_are_equal_false_equal_nan(tmp_path: Path) -> None:
    generator1 = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(
            path_uri=tmp_path.joinpath("uri/shards"),
            shards={
                "train": ShardTupleGenerator(
                    shard=JsonShardGenerator(
                        path_shard=tmp_path.joinpath("data/shards/train/shards"),
                        path_uri=tmp_path.joinpath("uri/shards/train/shards"),
                        data=DataGenerator([1, 2, 3, float("nan")]),
                    ),
                    num_shards=2,
                    path_uri=tmp_path.joinpath("uri/shards/train"),
                ),
            },
        ),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("uri/assets")),
    )
    generator2 = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(
            path_uri=tmp_path.joinpath("uri/shards"),
            shards={
                "train": ShardTupleGenerator(
                    shard=JsonShardGenerator(
                        path_shard=tmp_path.joinpath("data/shards/train/shards"),
                        path_uri=tmp_path.joinpath("uri/shards/train/shards"),
                        data=DataGenerator([1, 2, 3, float("nan")]),
                    ),
                    num_shards=2,
                    path_uri=tmp_path.joinpath("uri/shards/train"),
                ),
            },
        ),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("uri/assets")),
    )
    assert not objects_are_equal(generator1, generator2)
