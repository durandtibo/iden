from __future__ import annotations

from typing import TYPE_CHECKING

from iden.dataset import VanillaDataset
from iden.dataset.creator import VanillaDatasetCreator
from iden.shard import ShardDict
from iden.shard.creator import JsonShardCreator, ShardDictCreator, ShardTupleCreator

if TYPE_CHECKING:
    from pathlib import Path


###########################################
#     Tests for VanillaDatasetCreator     #
###########################################


def create_dataset_creator(path: Path) -> VanillaDatasetCreator:
    return VanillaDatasetCreator(
        path_uri=path,
        shards=ShardDictCreator(
            path_uri=path.joinpath("uri/shards"),
            shards={
                "train": ShardTupleCreator(
                    shard=JsonShardCreator(
                        path_shard=path.joinpath("data/shards/train/shards"),
                        path_uri=path.joinpath("uri/shards/train/shards"),
                        data=[1, 2, 3],
                    ),
                    num_shards=2,
                    path_uri=path.joinpath("uri/shards/train"),
                ),
                "val": ShardTupleCreator(
                    shard=JsonShardCreator(
                        path_shard=path.joinpath("data/shards/val/shards"),
                        path_uri=path.joinpath("uri/shards/val/shards"),
                        data=[4, 5, 6],
                    ),
                    num_shards=2,
                    path_uri=path.joinpath("uri/shards/val"),
                ),
            },
        ),
        assets=ShardDictCreator(shards={}, path_uri=path.joinpath("uri/assets")),
    )


def test_vanilla_dataset_creator_repr(tmp_path: Path) -> None:
    creator = create_dataset_creator(tmp_path)
    assert repr(creator).startswith("VanillaDatasetCreator(")


def test_vanilla_dataset_creator_str(tmp_path: Path) -> None:
    creator = create_dataset_creator(tmp_path)
    assert str(creator).startswith("VanillaDatasetCreator(")


def test_vanilla_dataset_creator_create(tmp_path: Path) -> None:
    creator = create_dataset_creator(tmp_path)
    dataset = creator.create("001111")
    assert dataset.equal(
        VanillaDataset(
            uri=tmp_path.joinpath("001111").as_uri(),
            shards=ShardDict.from_uri(tmp_path.joinpath("uri/shards/shards").as_uri()),
            assets=ShardDict.from_uri(tmp_path.joinpath("uri/assets/assets").as_uri()),
        ),
    )
