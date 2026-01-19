from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coola.equality.tester import get_default_registry
from objectory import OBJECT_TARGET

from iden.data.generator import (
    BaseDataGenerator,
    DataGenerator,
    is_data_generator_config,
    setup_data_generator,
)
from iden.shard.generator import JsonShardGenerator

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

##############################################
#     Tests for is_data_generator_config     #
##############################################


def test_is_data_generator_config_true() -> None:
    assert is_data_generator_config({OBJECT_TARGET: "iden.data.generator.DataGenerator"})


def test_is_data_generator_config_false() -> None:
    assert not is_data_generator_config({OBJECT_TARGET: "iden.shard.generator.JsonShardGenerator"})


##########################################
#     Tests for setup_data_generator     #
##########################################


def test_setup_data_generator_object() -> None:
    generator = DataGenerator([1, 2, 3])
    assert setup_data_generator(generator) is generator


def test_setup_data_generator_dict() -> None:
    assert isinstance(
        setup_data_generator(
            {OBJECT_TARGET: "iden.data.generator.DataGenerator", "data": [1, 2, 3]}
        ),
        DataGenerator,
    )


def test_setup_data_generator_incorrect_type(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_data_generator(
                {
                    OBJECT_TARGET: "iden.shard.generator.JsonShardGenerator",
                    "data": [1, 2, 3],
                    "path_uri": tmp_path.joinpath("uri"),
                    "path_shard": tmp_path.joinpath("data"),
                }
            ),
            JsonShardGenerator,
        )
        assert caplog.messages


def test_equality_tester_registry_has_equality_tester() -> None:
    assert get_default_registry().has_equality_tester(BaseDataGenerator)
