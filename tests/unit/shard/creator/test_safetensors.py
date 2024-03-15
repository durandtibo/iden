from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from coola import objects_are_equal
from coola.testing import torch_available
from coola.utils import is_torch_available

from iden.shard import TorchSafetensorsShard
from iden.shard.creator import TorchSafetensorsShardCreator
from iden.testing import safetensors_available

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if TYPE_CHECKING:
    from pathlib import Path

##################################################
#     Tests for TorchSafetensorsShardCreator     #
##################################################


@safetensors_available
@torch_available
def test_torch_safetensors_shard_creator_repr(tmp_path: Path) -> None:
    assert repr(
        TorchSafetensorsShardCreator(
            data={"key1": torch.ones(2, 3), "key2": torch.arange(5)},
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("TorchSafetensorsShardCreator(")


@safetensors_available
@torch_available
def test_torch_safetensors_shard_creator_str(tmp_path: Path) -> None:
    assert str(
        TorchSafetensorsShardCreator(
            data={"key1": torch.ones(2, 3), "key2": torch.arange(5)},
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("TorchSafetensorsShardCreator(")


@safetensors_available
@torch_available
def test_torch_safetensors_shard_creator_create(tmp_path: Path) -> None:
    creator = TorchSafetensorsShardCreator(
        data={"key1": torch.ones(2, 3), "key2": torch.arange(5)},
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    shard = creator.create("000001")
    assert shard.equal(
        TorchSafetensorsShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.safetensors"),
        )
    )
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


def test_torch_safetensors_shard_creator_no_safetensors(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(RuntimeError, match="`safetensors` package is required but not installed."),
    ):
        TorchSafetensorsShardCreator(
            data={"key1": torch.ones(2, 3), "key2": torch.arange(5)},
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )


def test_torch_safetensors_shard_creator_no_torch(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match="`torch` package is required but not installed."),
    ):
        TorchSafetensorsShardCreator(
            data={"key1": torch.ones(2, 3), "key2": torch.arange(5)},
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
