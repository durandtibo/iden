from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coola.equality.testers import EqualityTester
from objectory import OBJECT_TARGET

from iden.shard import JsonShard
from iden.shard.loader import (
    BaseShardLoader,
    JsonShardLoader,
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
    assert is_shard_loader_config({OBJECT_TARGET: "iden.shard.loader.JsonShardLoader"})


def test_is_shard_loader_config_false() -> None:
    assert not is_shard_loader_config({OBJECT_TARGET: "iden.shard.JsonShard"})


########################################
#     Tests for setup_shard_loader     #
########################################


def test_setup_shard_loader_object() -> None:
    loader = JsonShardLoader()
    assert setup_shard_loader(loader) is loader


def test_setup_shard_loader_dict() -> None:
    assert isinstance(
        setup_shard_loader({OBJECT_TARGET: "iden.shard.loader.JsonShardLoader"}),
        JsonShardLoader,
    )


def test_setup_shard_loader_incorrect_type(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_shard_loader(
                {OBJECT_TARGET: "iden.shard.JsonShard", "uri": "", "path": tmp_path}
            ),
            JsonShard,
        )
        assert caplog.messages


def test_equality_tester_has_comparator() -> None:
    assert EqualityTester.has_comparator(BaseShardLoader)
