from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from coola import objects_are_equal
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available

from iden.shard import (
    NumpySafetensorsShard,
    TorchSafetensorsShard,
    create_numpy_safetensors_shard,
    create_torch_safetensors_shard,
)
from iden.shard.loader import NumpySafetensorsShardLoader, TorchSafetensorsShardLoader
from iden.testing import safetensors_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_np(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.safetensors")


@pytest.fixture(scope="module")
def uri_np(tmp_path_factory: pytest.TempPathFactory, path_np: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_numpy_safetensors_shard(
        data={"key1": np.ones((2, 3)), "key2": np.arange(5)}, uri=uri_, path=path_np
    )
    return uri_


@pytest.fixture(scope="module")
def path_torch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.safetensors")


@pytest.fixture(scope="module")
def uri_torch(tmp_path_factory: pytest.TempPathFactory, path_torch: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_torch_safetensors_shard(
        data={"key1": torch.ones(2, 3), "key2": torch.arange(5)}, uri=uri_, path=path_torch
    )
    return uri_


#################################################
#     Tests for NumpySafetensorsShardLoader     #
#################################################


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_loader_str() -> None:
    assert str(NumpySafetensorsShardLoader()).startswith("NumpySafetensorsShardLoader(")


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_loader_load(uri_np: str, path_np: Path) -> None:
    shard = NumpySafetensorsShardLoader().load(uri_np)
    assert shard.equal(NumpySafetensorsShard(uri=uri_np, path=path_np))
    assert objects_are_equal(shard.get_data(), {"key1": np.ones((2, 3)), "key2": np.arange(5)})


def test_numpy_safetensors_shard_loader_no_safetensors() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(
            RuntimeError, match="`|'safetensors`|' package is required but not installed."
        ),
    ):
        NumpySafetensorsShardLoader()


def test_numpy_safetensors_shard_loader_no_numpy() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match="`|'numpy`|' package is required but not installed."),
    ):
        NumpySafetensorsShardLoader()


#################################################
#     Tests for TorchSafetensorsShardLoader     #
#################################################


@safetensors_available
@torch_available
def test_torch_safetensors_shard_loader_str() -> None:
    assert str(TorchSafetensorsShardLoader()).startswith("TorchSafetensorsShardLoader(")


@safetensors_available
@torch_available
def test_torch_safetensors_shard_loader_load(uri_torch: str, path_torch: Path) -> None:
    shard = TorchSafetensorsShardLoader().load(uri_torch)
    assert shard.equal(TorchSafetensorsShard(uri=uri_torch, path=path_torch))
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


def test_torch_safetensors_shard_loader_no_safetensors() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(
            RuntimeError, match="`|'safetensors`|' package is required but not installed."
        ),
    ):
        TorchSafetensorsShardLoader()


def test_torch_safetensors_shard_loader_no_torch() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match="`|'torch`|' package is required but not installed."),
    ):
        TorchSafetensorsShardLoader()
