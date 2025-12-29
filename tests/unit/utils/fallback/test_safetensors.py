from __future__ import annotations

from types import ModuleType

import pytest

from iden.utils.fallback.safetensors import safetensors


def test_safetensors_is_module_type() -> None:
    assert isinstance(safetensors, ModuleType)


def test_safetensors_module_name() -> None:
    assert safetensors.__name__ == "safetensors"


def test_safetensors_nested_module_access() -> None:
    assert hasattr(safetensors, "numpy")
    assert hasattr(safetensors, "torch")


def test_safetensors_numpy_load_file_exists() -> None:
    assert hasattr(safetensors.numpy, "load_file")


def test_safetensors_numpy_load_file_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'safetensors' package is required but not installed."):
        safetensors.numpy.load_file()


def test_safetensors_numpy_save_file_exists() -> None:
    assert hasattr(safetensors.numpy, "save_file")


def test_safetensors_numpy_save_file_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'safetensors' package is required but not installed."):
        safetensors.numpy.save_file()


def test_safetensors_torch_load_file_exists() -> None:
    assert hasattr(safetensors.torch, "load_file")


def test_safetensors_torch_load_file_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'safetensors' package is required but not installed."):
        safetensors.torch.load_file()


def test_safetensors_torch_save_file_exists() -> None:
    assert hasattr(safetensors.torch, "save_file")


def test_safetensors_torch_save_file_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'safetensors' package is required but not installed."):
        safetensors.torch.save_file()
