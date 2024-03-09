from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal

from iden.dataset import VanillaDataset
from iden.dataset.exceptions import AssetNotFoundError, SplitNotFoundError
from iden.dataset.vanilla import prepare_shards
from iden.shard import BaseShard, JsonShard, create_json_shard

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, Iterable[BaseShard]]:
    path = tmp_path_factory.mktemp("data")
    return {
        "train": [
            create_json_shard(
                data=[1, 2, 3],
                uri=path.joinpath("train/uri1").as_uri(),
                path=path.joinpath("train/data1.json"),
            ),
            create_json_shard(
                data=[4, 5, 6],
                uri=path.joinpath("train/uri2").as_uri(),
                path=path.joinpath("train/data2.json"),
            ),
            create_json_shard(
                data=[7, 8],
                uri=path.joinpath("train/uri3").as_uri(),
                path=path.joinpath("train/data3.json"),
            ),
        ],
        "val": [],
        "test": [
            create_json_shard(
                data=[10, 11, 12, 13, 14, 15],
                uri=path.joinpath("test/uri1").as_uri(),
                path=path.joinpath("test/data1.json"),
            ),
        ],
    }


@pytest.fixture(scope="module")
def dataset(shards: Mapping[str, Iterable[BaseShard]]) -> VanillaDataset:
    return VanillaDataset(shards=shards, assets={"mean": 42})


####################################
#     Tests for VanillaDataset     #
####################################


def test_vanilla_dataset_str(dataset: VanillaDataset) -> None:
    assert str(dataset).startswith("VanillaDataset(")


def test_vanilla_dataset_str_empty() -> None:
    assert str(VanillaDataset(shards={})).startswith("VanillaDataset(")


def test_vanilla_dataset_get_asset(dataset: VanillaDataset) -> None:
    assert dataset.get_asset("mean") == 42


def test_vanilla_dataset_get_asset_missing(dataset: VanillaDataset) -> None:
    with pytest.raises(AssetNotFoundError, match="asset 'missing' does not exist"):
        dataset.get_asset("missing")


def test_vanilla_dataset_has_asset_true(dataset: VanillaDataset) -> None:
    assert dataset.has_asset("mean")


def test_vanilla_dataset_has_asset_false(dataset: VanillaDataset) -> None:
    assert not dataset.has_asset("missing")


def test_vanilla_dataset_get_shards_train(
    dataset: VanillaDataset, shards: Mapping[str, Iterable[BaseShard]]
) -> None:
    assert dataset.get_shards("train") == tuple(shards["train"])


def test_vanilla_dataset_get_shards_empty(dataset: VanillaDataset) -> None:
    assert dataset.get_shards("val") == ()


def test_vanilla_dataset_get_shards_missing(dataset: VanillaDataset) -> None:
    with pytest.raises(SplitNotFoundError, match="split 'missing' does not exist"):
        dataset.get_shards("missing")


def test_vanilla_dataset_get_num_shards(dataset: VanillaDataset) -> None:
    assert dataset.get_num_shards("train") == 3
    assert dataset.get_num_shards("val") == 0
    assert dataset.get_num_shards("test") == 1


def test_vanilla_dataset_get_num_shards_missing(dataset: VanillaDataset) -> None:
    with pytest.raises(SplitNotFoundError, match="split 'missing' does not exist"):
        dataset.get_num_shards("missing")


def test_vanilla_dataset_get_splits(dataset: VanillaDataset) -> None:
    assert dataset.get_splits() == {"train", "val", "test"}


def test_vanilla_dataset_get_splits_empty() -> None:
    assert VanillaDataset(shards={}).get_splits() == set()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_vanilla_dataset_has_split_true(dataset: VanillaDataset, split: str) -> None:
    assert dataset.has_split(split)


def test_vanilla_dataset_has_split_false(dataset: VanillaDataset) -> None:
    assert not dataset.has_split("missing")


####################################
#     Tests for prepare_shards     #
####################################


def test_prepare_shards(shards: Mapping[str, Iterable[BaseShard]]) -> None:
    assert objects_are_equal(
        prepare_shards(shards), {split: tuple(shards[split]) for split in ["train", "val", "test"]}
    )


def test_prepare_shards_empty() -> None:
    assert prepare_shards({}) == {}


def test_prepare_shards_sort_shards(tmp_path: Path) -> None:
    shards = {
        "train": [
            create_json_shard(
                data=[4, 5, 6],
                uri=tmp_path.joinpath("train/uri2").as_uri(),
            ),
            create_json_shard(
                data=[7, 8],
                uri=tmp_path.joinpath("train/uri3").as_uri(),
            ),
            create_json_shard(
                data=[1, 2, 3],
                uri=tmp_path.joinpath("train/uri1").as_uri(),
            ),
        ],
    }
    assert objects_are_equal(
        prepare_shards(shards),
        {
            "train": (
                JsonShard.from_uri(uri=tmp_path.joinpath("train/uri1").as_uri()),
                JsonShard.from_uri(uri=tmp_path.joinpath("train/uri2").as_uri()),
                JsonShard.from_uri(uri=tmp_path.joinpath("train/uri3").as_uri()),
            ),
        },
        show_difference=True,
    )
