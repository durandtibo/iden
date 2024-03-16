from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from iden.dataset.creator import (
    VanillaDatasetCreator,
    is_dataset_creator_config,
    setup_dataset_creator,
)
from iden.shard.creator import JsonShardCreator
from tests.unit.dataset.creator.test_vanilla import create_dataset_creator

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

###############################################
#     Tests for is_dataset_creator_config     #
###############################################


def test_is_dataset_creator_config_true() -> None:
    assert is_dataset_creator_config({OBJECT_TARGET: "iden.dataset.creator.VanillaDatasetCreator"})


def test_is_dataset_creator_config_false() -> None:
    assert not is_dataset_creator_config({OBJECT_TARGET: "iden.shard.creator.JsonShardCreator"})


###########################################
#     Tests for setup_dataset_creator     #
###########################################


def test_setup_dataset_creator_object(tmp_path: Path) -> None:
    creator = create_dataset_creator(tmp_path)
    assert setup_dataset_creator(creator) is creator


def test_setup_dataset_creator_dict(tmp_path: Path) -> None:
    assert isinstance(
        setup_dataset_creator(
            {
                "_target_": "iden.dataset.creator.VanillaDatasetCreator",
                "path_uri": tmp_path.joinpath("uri"),
                "shards": {
                    "_target_": "iden.shard.creator.ShardDictCreator",
                    "path_uri": tmp_path.joinpath("uri/shards"),
                    "shards": {},
                },
                "assets": {
                    "_target_": "iden.shard.creator.ShardDictCreator",
                    "path_uri": tmp_path.joinpath("uri/assets"),
                    "shards": {},
                },
            }
        ),
        VanillaDatasetCreator,
    )


def test_setup_dataset_creator_incorrect_type(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_dataset_creator(
                {
                    OBJECT_TARGET: "iden.shard.creator.JsonShardCreator",
                    "data": [1, 2, 3],
                    "path_uri": tmp_path.joinpath("uri"),
                    "path_shard": tmp_path.joinpath("data"),
                }
            ),
            JsonShardCreator,
        )
        assert caplog.messages
