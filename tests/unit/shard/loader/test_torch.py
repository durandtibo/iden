from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from coola import objects_are_equal
from coola.testing import torch_available
from coola.utils import is_torch_available

from iden.shard import TorchShard, create_torch_shard
from iden.shard.loader import TorchShardLoader

if TYPE_CHECKING:
    from pathlib import Path


if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.safetensors")


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_torch_shard(
        data={"key1": torch.ones(2, 3), "key2": torch.arange(5)}, uri=uri_, path=path
    )
    return uri_


######################################
#     Tests for TorchShardLoader     #
######################################


@torch_available
def test_torch_shard_loader_str() -> None:
    assert str(TorchShardLoader()).startswith("TorchShardLoader(")


@torch_available
def test_torch_shard_loader_load(uri: str, path: Path) -> None:
    shard = TorchShardLoader().load(uri)
    assert shard.equal(TorchShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


def test_torch_shard_loader_no_torch() -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."),
    ):
        TorchShardLoader()
