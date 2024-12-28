from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from coola import objects_are_equal

from iden.shard import YamlShard, create_yaml_shard
from iden.shard.loader import YamlShardLoader
from iden.testing import yaml_available

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.yaml")


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_yaml_shard(data={"key1": [1, 2, 3], "key2": "abc"}, uri=uri_, path=path)
    return uri_


#####################################
#     Tests for YamlShardLoader     #
#####################################


@yaml_available
def test_yaml_shard_loader_str() -> None:
    assert str(YamlShardLoader()).startswith("YamlShardLoader(")


@yaml_available
def test_yaml_shard_loader_load(uri: str, path: Path) -> None:
    shard = YamlShardLoader().load(uri)
    assert shard.equal(YamlShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": [1, 2, 3], "key2": "abc"})


def test_torch_shard_loader_no_torch() -> None:
    with (
        patch("iden.utils.imports.is_yaml_available", lambda: False),
        pytest.raises(RuntimeError, match="'yaml' package is required but not installed."),
    ):
        YamlShardLoader()
