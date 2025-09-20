from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available

from iden.io import save_text
from iden.io.safetensors import NumpySaver, TorchSaver
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


################################
#     Tests for NumpySaver     #
################################


@safetensors_available
@numpy_available
def test_numpy_saver_str() -> None:
    assert str(NumpySaver()).startswith("NumpySafetensorsSaver(")


@safetensors_available
@numpy_available
def test_numpy_saver_equal_true() -> None:
    assert NumpySaver().equal(NumpySaver())


@safetensors_available
@numpy_available
def test_numpy_saver_equal_false() -> None:
    assert not NumpySaver().equal(42.0)


@safetensors_available
@numpy_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_numpy_saver_equal_nan(equal_nan: bool) -> None:
    assert NumpySaver().equal(NumpySaver(), equal_nan=equal_nan)


@safetensors_available
@numpy_available
def test_numpy_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    saver = NumpySaver()
    saver.save({"key1": np.ones((2, 3)), "key2": np.arange(5)}, path)
    assert path.is_file()


@safetensors_available
@numpy_available
def test_numpy_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    save_text("hello", path)
    saver = NumpySaver()
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        saver.save({"key1": np.ones((2, 3)), "key2": np.arange(5)}, path)


@safetensors_available
@numpy_available
def test_numpy_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    save_text("hello", path)
    saver = NumpySaver()
    saver.save({"key1": np.ones((2, 3)), "key2": np.arange(5)}, path, exist_ok=True)
    assert path.is_file()


@safetensors_available
@numpy_available
def test_numpy_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    path.mkdir(parents=True, exist_ok=True)
    saver = NumpySaver()
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        saver.save({"key1": np.ones((2, 3)), "key2": np.arange(5)}, path)


def test_numpy_saver_no_safetensors() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'safetensors' package is required but not installed."),
    ):
        NumpySaver()


def test_numpy_saver_no_numpy() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'numpy' package is required but not installed."),
    ):
        NumpySaver()


################################
#     Tests for TorchSaver     #
################################


@safetensors_available
@torch_available
def test_torch_saver_str() -> None:
    assert str(TorchSaver()).startswith("TorchSafetensorsSaver(")


@safetensors_available
@torch_available
def test_torch_saver_equal_true() -> None:
    assert TorchSaver().equal(TorchSaver())


@safetensors_available
@torch_available
def test_torch_saver_equal_false() -> None:
    assert not TorchSaver().equal(42.0)


@safetensors_available
@torch_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_torch_saver_equal_nan(equal_nan: bool) -> None:
    assert TorchSaver().equal(TorchSaver(), equal_nan=equal_nan)


@safetensors_available
@torch_available
def test_torch_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    saver = TorchSaver()
    saver.save({"key1": torch.ones(2, 3), "key2": torch.arange(5)}, path)
    assert path.is_file()


@safetensors_available
@torch_available
def test_torch_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    save_text("hello", path)
    saver = TorchSaver()
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        saver.save({"key1": torch.ones(2, 3), "key2": torch.arange(5)}, path)


@safetensors_available
@torch_available
def test_torch_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    save_text("hello", path)
    saver = TorchSaver()
    saver.save({"key1": torch.ones(2, 3), "key2": torch.arange(5)}, path, exist_ok=True)
    assert path.is_file()


@safetensors_available
@torch_available
def test_torch_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.safetensors")
    path.mkdir(parents=True, exist_ok=True)
    saver = TorchSaver()
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        saver.save({"key1": torch.ones(2, 3), "key2": torch.arange(5)}, path)


def test_torch_saver_no_safetensors() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'safetensors' package is required but not installed."),
    ):
        TorchSaver()


def test_torch_saver_no_torch() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."),
    ):
        TorchSaver()
