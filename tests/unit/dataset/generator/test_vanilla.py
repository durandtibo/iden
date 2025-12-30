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


def test_vanilla_dataset_generator_equal_true(tmp_path: Path) -> None:
    generator1 = create_dataset_generator(tmp_path)
    generator2 = create_dataset_generator(tmp_path)
    assert generator1.equal(generator2)


def test_vanilla_dataset_generator_equal_false_different_path_uri(tmp_path: Path) -> None:
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
    assert not generator1.equal(generator2)


def test_vanilla_dataset_generator_equal_false_different_shards(tmp_path: Path) -> None:
    generator1 = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(path_uri=tmp_path.joinpath("one/uri/shards"), shards={}),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("uri/assets")),
    )
    generator2 = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(path_uri=tmp_path.joinpath("two/uri/shards"), shards={}),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("uri/assets")),
    )
    assert not generator1.equal(generator2)


def test_vanilla_dataset_generator_equal_false_different_assets(tmp_path: Path) -> None:
    generator1 = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(path_uri=tmp_path.joinpath("uri/shards"), shards={}),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("one/uri/assets")),
    )
    generator2 = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(path_uri=tmp_path.joinpath("uri/shards"), shards={}),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("two/uri/assets")),
    )
    assert not generator1.equal(generator2)


def test_vanilla_dataset_generator_equal_false_different_type(tmp_path: Path) -> None:
    generator = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(path_uri=tmp_path.joinpath("uri/shards"), shards={}),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("one/uri/assets")),
    )
    assert not generator.equal(42)


def test_vanilla_dataset_generator_equal_false_different_type_child(tmp_path: Path) -> None:
    class Child(VanillaDatasetGenerator): ...

    generator1 = VanillaDatasetGenerator(
        path_uri=tmp_path,
        shards=ShardDictGenerator(path_uri=tmp_path.joinpath("uri/shards"), shards={}),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("one/uri/assets")),
    )
    generator2 = Child(
        path_uri=tmp_path,
        shards=ShardDictGenerator(path_uri=tmp_path.joinpath("uri/shards"), shards={}),
        assets=ShardDictGenerator(shards={}, path_uri=tmp_path.joinpath("two/uri/assets")),
    )
    assert not generator1.equal(generator2)


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
