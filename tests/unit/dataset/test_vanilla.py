from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal
from coola.utils.path import sanitize_path
from objectory import OBJECT_TARGET

from iden.constants import ASSETS, LOADER, SHARDS
from iden.dataset import VanillaDataset
from iden.dataset.exceptions import AssetNotFoundError, SplitNotFoundError
from iden.dataset.vanilla import check_shards, create_vanilla_dataset
from iden.io import JsonSaver
from iden.shard import (
    BaseShard,
    ShardDict,
    ShardTuple,
    create_json_shard,
    create_shard_dict,
    create_shard_tuple,
)

if TYPE_CHECKING:
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
def shards(tmp_path_factory: pytest.TempPathFactory) -> ShardDict[ShardTuple[BaseShard]]:
    path = tmp_path_factory.mktemp("shard")
    return create_shard_dict(
        {
            "train": create_shard_tuple(
                shards=[
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
                uri=path.joinpath("uri_train").as_uri(),
            ),
            "val": create_shard_tuple(shards=[], uri=path.joinpath("uri_val").as_uri()),
            "test": create_shard_tuple(
                shards=[
                    create_json_shard(
                        data=[10, 11, 12, 13, 14, 15],
                        uri=path.joinpath("test/uri1").as_uri(),
                        path=path.joinpath("test/data1.json"),
                    ),
                ],
                uri=path.joinpath("uri_test").as_uri(),
            ),
        },
        uri=path.joinpath("uri").as_uri(),
    )


@pytest.fixture(scope="module")
def uri(
    tmp_path_factory: pytest.TempPathFactory,
    shards: ShardDict[ShardTuple[BaseShard]],
    assets: ShardDict,
) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    JsonSaver().save(
        VanillaDataset.generate_uri_config(shards=shards, assets=assets), sanitize_path(uri_)
    )
    return uri_


@pytest.fixture(scope="module")
def dataset(
    uri: str, shards: ShardDict[ShardTuple[BaseShard]], assets: ShardDict
) -> VanillaDataset:
    return VanillaDataset(uri=uri, shards=shards, assets=assets)


####################################
#     Tests for VanillaDataset     #
####################################


def test_vanilla_dataset_repr(dataset: VanillaDataset) -> None:
    assert repr(dataset).startswith("VanillaDataset(")


def test_vanilla_dataset_str(dataset: VanillaDataset) -> None:
    assert str(dataset).startswith("VanillaDataset(")


def test_vanilla_dataset_equal_true(
    uri: str, shards: ShardDict[ShardTuple[BaseShard]], assets: ShardDict
) -> None:
    assert VanillaDataset(uri=uri, shards=shards, assets=assets).equal(
        VanillaDataset(uri=uri, shards=shards, assets=assets)
    )


def test_vanilla_dataset_equal_false_different_uri(
    uri: str, shards: ShardDict[ShardTuple[BaseShard]], assets: ShardDict
) -> None:
    assert not VanillaDataset(uri=uri, shards=shards, assets=assets).equal(
        VanillaDataset(uri=uri + "123", shards=shards, assets=assets)
    )


def test_vanilla_dataset_equal_false_different_shards(
    uri: str, shards: ShardDict[ShardTuple[BaseShard]], assets: ShardDict, tmp_path: Path
) -> None:
    assert not VanillaDataset(uri=uri, shards=shards, assets=assets).equal(
        VanillaDataset(
            uri=uri,
            shards=create_shard_dict(shards={}, uri=tmp_path.joinpath("uri").as_uri()),
            assets=assets,
        )
    )


def test_vanilla_dataset_equal_false_different_asset(
    uri: str, shards: ShardDict[ShardTuple[BaseShard]], assets: ShardDict, tmp_path: Path
) -> None:
    assert not VanillaDataset(uri=uri, shards=shards, assets=assets).equal(
        VanillaDataset(
            uri=uri,
            shards=shards,
            assets=create_shard_dict(shards={}, uri=tmp_path.joinpath("uri_asset").as_uri()),
        )
    )


def test_vanilla_dataset_equal_false_different_type(
    uri: str, shards: ShardDict[ShardTuple[BaseShard]], assets: ShardDict
) -> None:
    assert not VanillaDataset(uri=uri, shards=shards, assets=assets).equal("meow")


def test_vanilla_dataset_get_asset(dataset: VanillaDataset) -> None:
    assert objects_are_equal(dataset.get_asset("stats").get_data(), {"mean": 42})


def test_vanilla_dataset_get_asset_missing(dataset: VanillaDataset) -> None:
    with pytest.raises(AssetNotFoundError, match=r"asset 'missing' does not exist"):
        dataset.get_asset("missing")


def test_vanilla_dataset_has_asset_true(dataset: VanillaDataset) -> None:
    assert dataset.has_asset("stats")


def test_vanilla_dataset_has_asset_false(dataset: VanillaDataset) -> None:
    assert not dataset.has_asset("missing")


def test_vanilla_dataset_get_shards_train(
    dataset: VanillaDataset, shards: ShardDict[ShardTuple[BaseShard]]
) -> None:
    assert dataset.get_shards("train") == shards["train"].get_data()


def test_vanilla_dataset_get_shards_empty(dataset: VanillaDataset) -> None:
    assert dataset.get_shards("val") == ()


def test_vanilla_dataset_get_shards_missing(dataset: VanillaDataset) -> None:
    with pytest.raises(SplitNotFoundError, match=r"split 'missing' does not exist"):
        dataset.get_shards("missing")


def test_vanilla_dataset_get_num_shards(dataset: VanillaDataset) -> None:
    assert dataset.get_num_shards("train") == 3
    assert dataset.get_num_shards("val") == 0
    assert dataset.get_num_shards("test") == 1


def test_vanilla_dataset_get_num_shards_missing(dataset: VanillaDataset) -> None:
    with pytest.raises(SplitNotFoundError, match=r"split 'missing' does not exist"):
        dataset.get_num_shards("missing")


def test_vanilla_dataset_get_splits(dataset: VanillaDataset) -> None:
    assert dataset.get_splits() == {"train", "val", "test"}


def test_vanilla_dataset_get_splits_empty(uri: str, assets: ShardDict, tmp_path: Path) -> None:
    assert (
        VanillaDataset(
            uri=uri,
            shards=create_shard_dict(shards={}, uri=tmp_path.joinpath("uri").as_uri()),
            assets=assets,
        ).get_splits()
        == set()
    )


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_vanilla_dataset_has_split_true(dataset: VanillaDataset, split: str) -> None:
    assert dataset.has_split(split)


def test_vanilla_dataset_has_split_false(dataset: VanillaDataset) -> None:
    assert not dataset.has_split("missing")


def test_vanilla_dataset_get_uri(dataset: VanillaDataset, uri: str) -> None:
    assert dataset.get_uri() == uri


def test_vanilla_dataset_from_uri(
    uri: str,
    shards: ShardDict[ShardTuple[BaseShard]],
    assets: ShardDict,
) -> None:
    shard = VanillaDataset.from_uri(uri)
    assert shard.equal(VanillaDataset(uri=uri, shards=shards, assets=assets))


def test_vanilla_dataset_generate_uri_config(
    shards: ShardDict[ShardTuple[BaseShard]],
    assets: ShardDict,
) -> None:
    config = VanillaDataset.generate_uri_config(shards=shards, assets=assets)
    assert objects_are_equal(
        config,
        {
            LOADER: {OBJECT_TARGET: "iden.dataset.loader.VanillaDatasetLoader"},
            SHARDS: shards.get_uri(),
            ASSETS: assets.get_uri(),
        },
    )


############################################
#     Tests for create_vanilla_dataset     #
############################################


def test_create_vanilla_dataset(
    shards: ShardDict[ShardTuple[BaseShard]], assets: ShardDict, tmp_path: Path
) -> None:
    uri_file = tmp_path.joinpath("data/uri")
    dataset = create_vanilla_dataset(shards=shards, assets=assets, uri=uri_file.as_uri())
    assert uri_file.is_file()
    assert isinstance(dataset, VanillaDataset)


##################################
#     Tests for check_shards     #
##################################


def test_check_shards(shards: ShardDict[ShardTuple[BaseShard]]) -> None:
    check_shards(shards)


def test_check_shards_empty(tmp_path: Path) -> None:
    check_shards(create_shard_dict(shards={}, uri=tmp_path.joinpath("uri").as_uri()))


def test_check_shards_incorrect_order(tmp_path: Path) -> None:
    shards = create_shard_dict(
        {
            "train": create_shard_tuple(
                shards=[
                    create_json_shard(
                        data=[4, 5, 6],
                        uri=tmp_path.joinpath("train/uri2").as_uri(),
                        path=tmp_path.joinpath("train/data2.json"),
                    ),
                    create_json_shard(
                        data=[1, 2, 3],
                        uri=tmp_path.joinpath("train/uri1").as_uri(),
                        path=tmp_path.joinpath("train/data1.json"),
                    ),
                    create_json_shard(
                        data=[7, 8],
                        uri=tmp_path.joinpath("train/uri3").as_uri(),
                        path=tmp_path.joinpath("train/data3.json"),
                    ),
                ],
                uri=tmp_path.joinpath("uri_train").as_uri(),
            ),
        },
        uri=tmp_path.joinpath("uri").as_uri(),
    )

    with pytest.raises(
        RuntimeError, match=r"split 'train' is not sorted by ascending order of URIs"
    ):
        check_shards(shards)


def test_check_shards_incorrect_type(tmp_path: Path) -> None:
    shards = create_json_shard(
        data=[1, 2, 3],
        uri=tmp_path.joinpath("train/uri1").as_uri(),
        path=tmp_path.joinpath("train/data1.json"),
    )

    with pytest.raises(TypeError, match=r"Incorrect shard type:"):
        check_shards(shards)
