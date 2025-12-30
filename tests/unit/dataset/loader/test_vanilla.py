from __future__ import annotations

import pytest
from coola.utils.path import sanitize_path

from iden.dataset import VanillaDataset
from iden.dataset.loader import VanillaDatasetLoader
from iden.io import JsonSaver
from iden.shard import (
    BaseShard,
    ShardDict,
    ShardTuple,
    create_json_shard,
    create_shard_dict,
    create_shard_tuple,
)


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


##########################################
#     Tests for VanillaDatasetLoader     #
##########################################


def test_vanilla_dataset_loader_repr() -> None:
    assert str(VanillaDatasetLoader()).startswith("VanillaDatasetLoader(")


def test_vanilla_dataset_loader_str() -> None:
    assert str(VanillaDatasetLoader()).startswith("VanillaDatasetLoader(")


def test_vanilla_dataset_loader_equal_true() -> None:
    assert VanillaDatasetLoader().equal(VanillaDatasetLoader())


def test_vanilla_dataset_loader_equal_false_different_type() -> None:
    assert not VanillaDatasetLoader().equal(42)


def test_vanilla_dataset_loader_equal_false_different_type_child() -> None:
    class Child(VanillaDatasetLoader): ...

    assert not VanillaDatasetLoader().equal(Child())


def test_vanilla_dataset_loader_load(uri: str, dataset: VanillaDataset) -> None:
    dataset = VanillaDatasetLoader().load(uri)
    assert dataset.equal(dataset)
