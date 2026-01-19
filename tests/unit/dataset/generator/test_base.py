from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coola.equality.tester import get_default_registry
from objectory import OBJECT_TARGET

from iden.dataset.generator import (
    BaseDatasetGenerator,
    VanillaDatasetGenerator,
    is_dataset_generator_config,
    setup_dataset_generator,
)
from iden.shard.generator import JsonShardGenerator
from tests.unit.dataset.generator.test_vanilla import create_dataset_generator

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

#################################################
#     Tests for is_dataset_generator_config     #
#################################################


def test_is_dataset_generator_config_true() -> None:
    assert is_dataset_generator_config(
        {OBJECT_TARGET: "iden.dataset.generator.VanillaDatasetGenerator"}
    )


def test_is_dataset_generator_config_false() -> None:
    assert not is_dataset_generator_config(
        {OBJECT_TARGET: "iden.shard.generator.JsonShardGenerator"}
    )


#############################################
#     Tests for setup_dataset_generator     #
#############################################


def test_setup_dataset_generator_object(tmp_path: Path) -> None:
    generator = create_dataset_generator(tmp_path)
    assert setup_dataset_generator(generator) is generator


def test_setup_dataset_generator_dict(tmp_path: Path) -> None:
    assert isinstance(
        setup_dataset_generator(
            {
                "_target_": "iden.dataset.generator.VanillaDatasetGenerator",
                "path_uri": tmp_path.joinpath("uri"),
                "shards": {
                    "_target_": "iden.shard.generator.ShardDictGenerator",
                    "path_uri": tmp_path.joinpath("uri/shards"),
                    "shards": {},
                },
                "assets": {
                    "_target_": "iden.shard.generator.ShardDictGenerator",
                    "path_uri": tmp_path.joinpath("uri/assets"),
                    "shards": {},
                },
            }
        ),
        VanillaDatasetGenerator,
    )


def test_setup_dataset_generator_incorrect_type(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_dataset_generator(
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
    assert get_default_registry().has_equality_tester(BaseDatasetGenerator)
