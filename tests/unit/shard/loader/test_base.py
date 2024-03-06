from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from iden.shard import PickleShard
from iden.shard.loader import (
    PickleShardLoader,
    is_shard_loader_config,
    setup_shard_loader,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

############################################
#     Tests for is_shard_loader_config     #
############################################


def test_is_shard_loader_config_true() -> None:
    assert is_shard_loader_config({OBJECT_TARGET: "iden.shard.loader.PickleShardLoader"})


def test_is_shard_loader_config_false() -> None:
    assert not is_shard_loader_config({OBJECT_TARGET: "iden.shard.PickleShard"})


########################################
#     Tests for setup_shard_loader     #
########################################


def test_setup_shard_loader_object() -> None:
    loader = PickleShardLoader()
    assert setup_shard_loader(loader) is loader


def test_setup_shard_loader_dict() -> None:
    assert isinstance(
        setup_shard_loader({OBJECT_TARGET: "iden.shard.loader.PickleShardLoader"}),
        PickleShardLoader,
    )


def test_setup_shard_loader_incorrect_type(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_shard_loader(
                {OBJECT_TARGET: "iden.shard.PickleShard", "uri": "", "path": tmp_path}
            ),
            PickleShard,
        )
        assert caplog.messages
