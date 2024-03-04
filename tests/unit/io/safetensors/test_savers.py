from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available

from iden.io.safetensors import NumpySafetensorsSaver, TorchSafetensorsSaver
from iden.testing import safetensors_available
from iden.utils.io import save_text

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


###########################################
#     Tests for NumpySafetensorsSaver     #
###########################################


@safetensors_available
@numpy_available
def test_numpy_safetensors_saver_str() -> None:
    assert str(NumpySafetensorsSaver()).startswith("NumpySafetensorsSaver(")


@safetensors_available
@numpy_available
def test_numpy_safetensors_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    saver = NumpySafetensorsSaver()
    saver.save({"key1": np.ones((2, 3)), "key2": np.arange(5)}, path)
    assert path.is_file()


@safetensors_available
@numpy_available
def test_numpy_safetensors_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    save_text("hello", path)
    saver = NumpySafetensorsSaver()
    with pytest.raises(FileExistsError, match="path .* already exists."):
        saver.save({"key1": np.ones((2, 3)), "key2": np.arange(5)}, path)


@safetensors_available
@numpy_available
def test_numpy_safetensors_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    save_text("hello", path)
    saver = NumpySafetensorsSaver()
    saver.save({"key1": np.ones((2, 3)), "key2": np.arange(5)}, path, exist_ok=True)
    assert path.is_file()


@safetensors_available
@numpy_available
def test_numpy_safetensors_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    path.mkdir(parents=True, exist_ok=True)
    saver = NumpySafetensorsSaver()
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        saver.save({"key1": np.ones((2, 3)), "key2": np.arange(5)}, path)


def test_numpy_safetensors_saver_no_safetensors() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(RuntimeError, match="`safetensors` package is required but not installed."),
    ):
        NumpySafetensorsSaver()


def test_numpy_safetensors_saver_no_numpy() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match="`numpy` package is required but not installed."),
    ):
        NumpySafetensorsSaver()


###########################################
#     Tests for TorchSafetensorsSaver     #
###########################################


@safetensors_available
@torch_available
def test_torch_safetensors_saver_str() -> None:
    assert str(TorchSafetensorsSaver()).startswith("TorchSafetensorsSaver(")


@safetensors_available
@torch_available
def test_torch_safetensors_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    saver = TorchSafetensorsSaver()
    saver.save({"key1": torch.ones(2, 3), "key2": torch.arange(5)}, path)
    assert path.is_file()


@safetensors_available
@torch_available
def test_torch_safetensors_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    save_text("hello", path)
    saver = TorchSafetensorsSaver()
    with pytest.raises(FileExistsError, match="path .* already exists."):
        saver.save({"key1": torch.ones(2, 3), "key2": torch.arange(5)}, path)


@safetensors_available
@torch_available
def test_torch_safetensors_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    save_text("hello", path)
    saver = TorchSafetensorsSaver()
    saver.save({"key1": torch.ones(2, 3), "key2": torch.arange(5)}, path, exist_ok=True)
    assert path.is_file()


@safetensors_available
@torch_available
def test_torch_safetensors_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    path.mkdir(parents=True, exist_ok=True)
    saver = TorchSafetensorsSaver()
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        saver.save({"key1": torch.ones(2, 3), "key2": torch.arange(5)}, path)


def test_torch_safetensors_saver_no_safetensors() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(RuntimeError, match="`safetensors` package is required but not installed."),
    ):
        TorchSafetensorsSaver()


def test_torch_safetensors_saver_no_torch() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match="`torch` package is required but not installed."),
    ):
        TorchSafetensorsSaver()
