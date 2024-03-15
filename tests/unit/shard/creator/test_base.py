from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from iden.shard import JsonShard
from iden.shard.creator import (
    JsonShardCreator,
    is_shard_creator_config,
    setup_shard_creator,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

#############################################
#     Tests for is_shard_creator_config     #
#############################################


def test_is_shard_creator_config_true() -> None:
    assert is_shard_creator_config({OBJECT_TARGET: "iden.shard.creator.JsonShardCreator"})


def test_is_shard_creator_config_false() -> None:
    assert not is_shard_creator_config({OBJECT_TARGET: "iden.shard.PickleShard"})


#########################################
#     Tests for setup_shard_creator     #
#########################################


def test_setup_shard_creator_object(tmp_path: Path) -> None:
    creator = JsonShardCreator(
        data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
    )
    assert setup_shard_creator(creator) is creator


def test_setup_shard_creator_dict(tmp_path: Path) -> None:
    assert isinstance(
        setup_shard_creator(
            {
                OBJECT_TARGET: "iden.shard.creator.JsonShardCreator",
                "data": [1, 2, 3],
                "path_uri": tmp_path.joinpath("uri"),
                "path_shard": tmp_path.joinpath("data"),
            }
        ),
        JsonShardCreator,
    )


def test_setup_shard_creator_incorrect_type(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_shard_creator(
                {OBJECT_TARGET: "iden.shard.JsonShard", "uri": "", "path": tmp_path}
            ),
            JsonShard,
        )
        assert caplog.messages
