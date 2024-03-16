from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from coola import objects_are_equal

from iden.data.generator import DataGenerator
from iden.shard import YamlShard
from iden.shard.generator import YamlShardGenerator
from iden.testing import yaml_available

if TYPE_CHECKING:
    from pathlib import Path

########################################
#     Tests for YamlShardGenerator     #
########################################


@yaml_available
def test_yaml_shard_generator_repr(tmp_path: Path) -> None:
    assert repr(
        YamlShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("YamlShardGenerator(")


@yaml_available
def test_yaml_shard_generator_str(tmp_path: Path) -> None:
    assert str(
        YamlShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("YamlShardGenerator(")


@yaml_available
def test_yaml_shard_generator_generate(tmp_path: Path) -> None:
    generator = YamlShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    shard = generator.generate("000001")
    assert shard.equal(
        YamlShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.yaml"),
        )
    )
    assert objects_are_equal(shard.get_data(), [1, 2, 3])


def test_yaml_shard_generator_no_yaml(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_yaml_available", lambda: False),
        pytest.raises(RuntimeError, match="`yaml` package is required but not installed."),
    ):
        YamlShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
