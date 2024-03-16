from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from coola import objects_are_equal
from coola.testing import torch_available

from iden.shard import TorchShard
from iden.shard.generator import TorchShardGenerator

if TYPE_CHECKING:
    from pathlib import Path

#########################################
#     Tests for TorchShardGenerator     #
#########################################


@torch_available
def test_torch_shard_generator_repr(tmp_path: Path) -> None:
    assert repr(
        TorchShardGenerator(
            data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
        )
    ).startswith("TorchShardGenerator(")


@torch_available
def test_torch_shard_generator_str(tmp_path: Path) -> None:
    assert str(
        TorchShardGenerator(
            data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
        )
    ).startswith("TorchShardGenerator(")


@torch_available
def test_torch_shard_generator_generate(tmp_path: Path) -> None:
    generator = TorchShardGenerator(
        data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
    )
    shard = generator.generate("000001")
    assert shard.equal(
        TorchShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.pt"),
        )
    )
    assert objects_are_equal(shard.get_data(), [1, 2, 3])


def test_torch_shard_generator_no_torch(tmp_path: Path) -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match="`torch` package is required but not installed."),
    ):
        TorchShardGenerator(
            data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
        )
