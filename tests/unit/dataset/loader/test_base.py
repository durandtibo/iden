from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from coola.equality.testers import EqualityTester
from coola.utils.path import sanitize_path
from objectory import OBJECT_TARGET

from iden.dataset import VanillaDataset
from iden.dataset.loader import (
    BaseDatasetLoader,
    VanillaDatasetLoader,
    is_dataset_loader_config,
    setup_dataset_loader,
)
from iden.io import JsonSaver
from iden.shard import (
    BaseShard,
    PickleShard,
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


##############################################
#     Tests for is_dataset_loader_config     #
##############################################


def test_is_dataset_loader_config_true() -> None:
    assert is_dataset_loader_config({OBJECT_TARGET: "iden.dataset.loader.VanillaDatasetLoader"})


def test_is_dataset_loader_config_false() -> None:
    assert not is_dataset_loader_config({OBJECT_TARGET: "iden.dataset.VanillaDataset"})


########################################
#     Tests for setup_dataset_loader     #
########################################


def test_setup_dataset_loader_object() -> None:
    loader = VanillaDatasetLoader()
    assert setup_dataset_loader(loader) is loader


def test_setup_dataset_loader_dict() -> None:
    assert isinstance(
        setup_dataset_loader({OBJECT_TARGET: "iden.dataset.loader.VanillaDatasetLoader"}),
        VanillaDatasetLoader,
    )


def test_setup_dataset_loader_incorrect_type(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_dataset_loader(
                {OBJECT_TARGET: "iden.shard.PickleShard", "uri": "", "path": tmp_path}
            ),
            PickleShard,
        )
        assert caplog.messages


def test_equality_tester_has_comparator() -> None:
    assert EqualityTester.has_comparator(BaseDatasetLoader)
