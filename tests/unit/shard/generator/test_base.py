from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coola.equality.testers import EqualityTester
from objectory import OBJECT_TARGET

from iden.data.generator import DataGenerator
from iden.shard import JsonShard
from iden.shard.generator import (
    BaseShardGenerator,
    JsonShardGenerator,
    is_shard_generator_config,
    setup_shard_generator,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

###############################################
#     Tests for is_shard_generator_config     #
###############################################


def test_is_shard_generator_config_true() -> None:
    assert is_shard_generator_config({OBJECT_TARGET: "iden.shard.generator.JsonShardGenerator"})


def test_is_shard_generator_config_false() -> None:
    assert not is_shard_generator_config(
        {OBJECT_TARGET: "iden.dataset.generator.VanillaDatasetGenerator"}
    )


###########################################
#     Tests for setup_shard_generator     #
###########################################


def test_setup_shard_generator_object(tmp_path: Path) -> None:
    generator = JsonShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert setup_shard_generator(generator) is generator


def test_setup_shard_generator_dict(tmp_path: Path) -> None:
    assert isinstance(
        setup_shard_generator(
            {
                OBJECT_TARGET: "iden.shard.generator.JsonShardGenerator",
                "data": [1, 2, 3],
                "path_uri": tmp_path.joinpath("uri"),
                "path_shard": tmp_path.joinpath("data"),
            }
        ),
        JsonShardGenerator,
    )


def test_setup_shard_generator_incorrect_type(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_shard_generator(
                {OBJECT_TARGET: "iden.shard.JsonShard", "uri": "", "path": tmp_path}
            ),
            JsonShard,
        )
        assert caplog.messages


def test_equality_tester_has_comparator() -> None:
    assert EqualityTester.has_comparator(BaseShardGenerator)
