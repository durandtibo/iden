from __future__ import annotations

from typing import TYPE_CHECKING

from iden.data.generator import DataGenerator
from iden.dataset import VanillaDataset
from iden.dataset.generator import VanillaDatasetGenerator
from iden.shard import ShardDict
from iden.shard.generator import (
    JsonShardGenerator,
    ShardDictGenerator,
    ShardTupleGenerator,
)

if TYPE_CHECKING:
    from pathlib import Path


#############################################
#     Tests for VanillaDatasetGenerator     #
#############################################


def create_dataset_generator(path: Path) -> VanillaDatasetGenerator:
    return VanillaDatasetGenerator(
        path_uri=path,
        shards=ShardDictGenerator(
            path_uri=path.joinpath("uri/shards"),
            shards={
                "train": ShardTupleGenerator(
                    shard=JsonShardGenerator(
                        path_shard=path.joinpath("data/shards/train/shards"),
                        path_uri=path.joinpath("uri/shards/train/shards"),
                        data=DataGenerator([1, 2, 3]),
                    ),
                    num_shards=2,
                    path_uri=path.joinpath("uri/shards/train"),
                ),
                "val": ShardTupleGenerator(
                    shard=JsonShardGenerator(
                        path_shard=path.joinpath("data/shards/val/shards"),
                        path_uri=path.joinpath("uri/shards/val/shards"),
                        data=DataGenerator([4, 5, 6]),
                    ),
                    num_shards=2,
                    path_uri=path.joinpath("uri/shards/val"),
                ),
            },
        ),
        assets=ShardDictGenerator(shards={}, path_uri=path.joinpath("uri/assets")),
    )


def test_vanilla_dataset_generator_repr(tmp_path: Path) -> None:
    generator = create_dataset_generator(tmp_path)
    assert repr(generator).startswith("VanillaDatasetGenerator(")


def test_vanilla_dataset_generator_str(tmp_path: Path) -> None:
    generator = create_dataset_generator(tmp_path)
    assert str(generator).startswith("VanillaDatasetGenerator(")


def test_vanilla_dataset_generator_generate(tmp_path: Path) -> None:
    generator = create_dataset_generator(tmp_path)
    dataset = generator.generate("001111")
    assert dataset.equal(
        VanillaDataset(
            uri=tmp_path.joinpath("001111").as_uri(),
            shards=ShardDict.from_uri(tmp_path.joinpath("uri/shards/shards").as_uri()),
            assets=ShardDict.from_uri(tmp_path.joinpath("uri/assets/assets").as_uri()),
        ),
    )
