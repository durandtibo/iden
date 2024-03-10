from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal
from objectory import OBJECT_TARGET

from iden.constants import LOADER, SHARDS
from iden.dataset import VanillaDataset
from iden.dataset.exceptions import AssetNotFoundError, SplitNotFoundError
from iden.dataset.vanilla import create_vanilla_dataset, prepare_shards
from iden.io import JsonSaver
from iden.shard import (
    BaseShard,
    JsonShard,
    ShardDict,
    create_json_shard,
    create_shard_dict,
)
from iden.utils.path import sanitize_path

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path


@pytest.fixture(scope="module")
def assets(tmp_path_factory: pytest.TempPathFactory) -> ShardDict:
    path = tmp_path_factory.mktemp("asset")
    stats = create_json_shard(
        data={"mean": 42},
        uri=path.joinpath("uri_stats").as_uri(),
        path=path.joinpath("data_stats.json"),
    )
    return create_shard_dict(shards={"stats": stats}, uri=path.joinpath("uri_asset").as_uri())


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, Iterable[BaseShard]]:
    path = tmp_path_factory.mktemp("shard")
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
def uri(tmp_path_factory: pytest.TempPathFactory, shards: Mapping[str, Iterable[BaseShard]]) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    JsonSaver().save(VanillaDataset.generate_uri_config(shards), sanitize_path(uri_))
    return uri_


@pytest.fixture(scope="module")
def dataset(
    uri: str, shards: Mapping[str, Iterable[BaseShard]], assets: ShardDict
) -> VanillaDataset:
    return VanillaDataset(uri=uri, shards=shards, assets=assets)


####################################
#     Tests for VanillaDataset     #
####################################


def test_vanilla_dataset_repr(dataset: VanillaDataset) -> None:
    assert repr(dataset).startswith("VanillaDataset(")


def test_vanilla_dataset_repr_empty(uri: str, tmp_path: Path) -> None:
    assert repr(
        VanillaDataset(
            uri=uri,
            shards={},
            assets=create_shard_dict(shards={}, uri=tmp_path.joinpath("uri_asset").as_uri()),
        )
    ).startswith("VanillaDataset(")


def test_vanilla_dataset_str(dataset: VanillaDataset) -> None:
    assert str(dataset).startswith("VanillaDataset(")


def test_vanilla_dataset_str_empty(uri: str, tmp_path: Path) -> None:
    assert str(
        VanillaDataset(
            uri=uri,
            shards={},
            assets=create_shard_dict(shards={}, uri=tmp_path.joinpath("uri_asset").as_uri()),
        )
    ).startswith("VanillaDataset(")


def test_vanilla_dataset_equal_true(
    uri: str, shards: Mapping[str, Iterable[BaseShard]], assets: ShardDict
) -> None:
    assert VanillaDataset(uri=uri, shards=shards, assets=assets).equal(
        VanillaDataset(uri=uri, shards=shards, assets=assets)
    )


def test_vanilla_dataset_equal_false_different_uri(
    uri: str, shards: Mapping[str, Iterable[BaseShard]], assets: ShardDict
) -> None:
    assert not VanillaDataset(uri=uri, shards=shards, assets=assets).equal(
        VanillaDataset(uri=uri + "123", shards=shards, assets=assets)
    )


def test_vanilla_dataset_equal_false_different_shards(
    uri: str, shards: Mapping[str, Iterable[BaseShard]], assets: ShardDict
) -> None:
    assert not VanillaDataset(uri=uri, shards=shards, assets=assets).equal(
        VanillaDataset(uri=uri, shards={}, assets=assets)
    )


def test_vanilla_dataset_equal_false_different_asset(
    uri: str, shards: Mapping[str, Iterable[BaseShard]], assets: ShardDict, tmp_path: Path
) -> None:
    assert not VanillaDataset(uri=uri, shards=shards, assets=assets).equal(
        VanillaDataset(
            uri=uri,
            shards=shards,
            assets=create_shard_dict(shards={}, uri=tmp_path.joinpath("uri_asset").as_uri()),
        )
    )


def test_vanilla_dataset_equal_false_different_type(
    uri: str, shards: Mapping[str, Iterable[BaseShard]], assets: ShardDict
) -> None:
    assert not VanillaDataset(uri=uri, shards=shards, assets=assets).equal("meow")


def test_vanilla_dataset_get_asset(dataset: VanillaDataset) -> None:
    assert objects_are_equal(dataset.get_asset("stats").get_data(), {"mean": 42})


def test_vanilla_dataset_get_asset_missing(dataset: VanillaDataset) -> None:
    with pytest.raises(AssetNotFoundError, match="asset 'missing' does not exist"):
        dataset.get_asset("missing")


def test_vanilla_dataset_has_asset_true(dataset: VanillaDataset) -> None:
    assert dataset.has_asset("stats")


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


def test_vanilla_dataset_get_splits_empty(uri: str, assets: ShardDict) -> None:
    assert VanillaDataset(uri=uri, shards={}, assets=assets).get_splits() == set()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_vanilla_dataset_has_split_true(dataset: VanillaDataset, split: str) -> None:
    assert dataset.has_split(split)


def test_vanilla_dataset_has_split_false(dataset: VanillaDataset) -> None:
    assert not dataset.has_split("missing")


def test_vanilla_dataset_get_uri(dataset: VanillaDataset, uri: str) -> None:
    assert dataset.get_uri() == uri


def test_vanilla_dataset_generate_uri_config(shards: Mapping[str, Sequence[BaseShard]]) -> None:
    config = VanillaDataset.generate_uri_config(shards)
    assert objects_are_equal(
        config,
        {
            LOADER: {OBJECT_TARGET: "iden.dataset.loader.VanillaDatasetLoader"},
            SHARDS: {
                "train": (
                    shards["train"][0].get_uri(),
                    shards["train"][1].get_uri(),
                    shards["train"][2].get_uri(),
                ),
                "val": (),
                "test": (shards["test"][0].get_uri(),),
            },
        },
    )


############################################
#     Tests for create_vanilla_dataset     #
############################################


def test_create_vanilla_dataset(
    shards: Mapping[str, Iterable[BaseShard]], assets: ShardDict, tmp_path: Path
) -> None:
    uri_file = tmp_path.joinpath("data/uri")
    dataset = create_vanilla_dataset(shards=shards, assets=assets, uri=uri_file.as_uri())
    assert uri_file.is_file()
    assert isinstance(dataset, VanillaDataset)


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
