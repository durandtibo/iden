from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from coola import objects_are_equal
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available

from iden.io.safetensors import NumpyLoader, NumpySaver, TorchLoader, TorchSaver
from iden.testing import safetensors_available

if TYPE_CHECKING:
    from pathlib import Path

if is_numpy_available():
    import numpy as np
else:
    np = Mock()  # pragma: no cover

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


@pytest.fixture(scope="module")
def path_numpy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.safetensors")
    NumpySaver().save({"key1": np.ones((2, 3)), "key2": np.arange(5)}, path)
    return path


@pytest.fixture(scope="module")
def path_torch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.safetensors")
    TorchSaver().save({"key1": torch.ones(2, 3), "key2": torch.arange(5)}, path)
    return path


#################################
#     Tests for NumpyLoader     #
#################################


@safetensors_available
@numpy_available
def test_numpy_loader_str() -> None:
    assert str(NumpyLoader()).startswith("NumpySafetensorsLoader(")


@safetensors_available
@numpy_available
def test_numpy_loader_load(path_numpy: Path) -> None:
    data = NumpyLoader().load(path_numpy)
    assert objects_are_equal(data, {"key1": np.ones((2, 3)), "key2": np.arange(5)})


def test_numpy_loader_no_safetensors() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(
            RuntimeError, match="`|'safetensors`|' package is required but not installed."
        ),
    ):
        NumpyLoader()


def test_numpy_loader_no_numpy() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match="`|'numpy`|' package is required but not installed."),
    ):
        NumpyLoader()


#################################
#     Tests for TorchLoader     #
#################################


@safetensors_available
@torch_available
def test_torch_loader_str() -> None:
    assert str(TorchLoader()).startswith("TorchSafetensorsLoader(")


@safetensors_available
@torch_available
def test_torch_loader_save(path_torch: Path) -> None:
    data = TorchLoader().load(path_torch)
    assert objects_are_equal(data, {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


def test_torch_loader_no_safetensors() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(
            RuntimeError, match="`|'safetensors`|' package is required but not installed."
        ),
    ):
        TorchLoader()


def test_torch_loader_no_torch() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match="`|'torch`|' package is required but not installed."),
    ):
        TorchLoader()
