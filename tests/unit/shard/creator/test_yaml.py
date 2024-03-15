from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from coola import objects_are_equal

from iden.shard import YamlShard
from iden.shard.creator import YamlShardCreator
from iden.testing import yaml_available

if TYPE_CHECKING:
    from pathlib import Path

###############################
#     Tests for YamlShard     #
###############################


@yaml_available
def test_yaml_shard_creator_repr(tmp_path: Path) -> None:
    assert repr(
        YamlShardCreator(
            data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
        )
    ).startswith("YamlShardCreator(")


@yaml_available
def test_yaml_shard_creator_str(tmp_path: Path) -> None:
    assert str(
        YamlShardCreator(
            data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
        )
    ).startswith("YamlShardCreator(")


@yaml_available
def test_yaml_shard_creator_create(tmp_path: Path) -> None:
    creator = YamlShardCreator(
        data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
    )
    shard = creator.create("000001")
    assert shard.equal(
        YamlShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.yaml"),
        )
    )
    assert objects_are_equal(shard.get_data(), [1, 2, 3])


def test_yaml_shard_creator_no_yaml(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_yaml_available", lambda: False),
        pytest.raises(RuntimeError, match="`yaml` package is required but not installed."),
    ):
        YamlShardCreator(
            data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
        )
