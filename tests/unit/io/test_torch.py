from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from coola import objects_are_equal
from coola.testing import torch_available
from coola.utils import is_torch_available

from iden.io import TorchLoader, TorchSaver, load_torch, save_torch
from iden.io.torch import get_loader_mapping
from tests.conftest import torch_greater_equal_1_13

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_torch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.pt")
    save_torch({"key1": [1, 2, 3], "key2": "abc", "key3": torch.arange(5)}, path)
    return path


#################################
#     Tests for TorchLoader     #
#################################


@torch_available
def test_torch_loader_repr() -> None:
    assert repr(TorchLoader()) == "TorchLoader()"


@torch_available
def test_torch_loader_repr_with_kwargs() -> None:
    assert repr(TorchLoader(weights_only=True)) == "TorchLoader(weights_only=True)"


@torch_available
def test_torch_loader_str() -> None:
    assert str(TorchLoader()) == "TorchLoader()"


@torch_available
def test_torch_loader_str_with_kwargs() -> None:
    assert str(TorchLoader(weights_only=True)) == "TorchLoader(weights_only=True)"


@torch_available
def test_torch_loader_eq_true() -> None:
    assert TorchLoader() == TorchLoader()


@torch_available
def test_torch_loader_eq_false_different_kwargs() -> None:
    assert TorchLoader(weights_only=True) != TorchLoader(weights_only=False)


@torch_available
def test_torch_loader_eq_false_different_type() -> None:
    assert TorchLoader() != TorchSaver()


@torch_available
def test_torch_loader_equal_true() -> None:
    assert TorchLoader().equal(TorchLoader())


@torch_available
def test_torch_loader_equal_false_different_kwargs() -> None:
    assert not TorchLoader(weights_only=True).equal(TorchLoader(weights_only=False))


@torch_available
def test_torch_loader_equal_false_different_type() -> None:
    assert not TorchLoader().equal(TorchSaver())


@torch_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_torch_loader_equal_nan(equal_nan: bool) -> None:
    assert TorchLoader().equal(TorchLoader(), equal_nan=equal_nan)


@torch_available
def test_torch_loader_load(path_torch: Path) -> None:
    assert objects_are_equal(
        TorchLoader().load(path_torch),
        {
            "key1": [1, 2, 3],
            "key2": "abc",
            "key3": torch.arange(5),
        },
    )


@torch_available
@torch_greater_equal_1_13
def test_torch_loader_load_weights_only_false(path_torch: Path) -> None:
    assert objects_are_equal(
        TorchLoader(weights_only=False).load(path_torch),
        {
            "key1": [1, 2, 3],
            "key2": "abc",
            "key3": torch.arange(5),
        },
    )


def test_torch_loader_no_torch() -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."),
    ):
        TorchLoader()


################################
#     Tests for TorchSaver     #
################################


@torch_available
def test_torch_saver_repr() -> None:
    assert repr(TorchSaver()) == "TorchSaver()"


@torch_available
def test_torch_saver_repr_with_kwargs() -> None:
    assert repr(TorchSaver(pickle_protocol=5)) == "TorchSaver(pickle_protocol=5)"


@torch_available
def test_torch_saver_str() -> None:
    assert str(TorchSaver()) == "TorchSaver()"


@torch_available
def test_torch_saver_str_with_kwargs() -> None:
    assert str(TorchSaver(pickle_protocol=5)) == "TorchSaver(pickle_protocol=5)"


@torch_available
def test_torch_saver_eq_true() -> None:
    assert TorchSaver() == TorchSaver()


@torch_available
def test_torch_saver_eq_false_different_kwargs() -> None:
    assert TorchSaver(pickle_protocol=5) != TorchSaver(pickle_protocol=4)


@torch_available
def test_torch_saver_eq_false_different_type() -> None:
    assert TorchSaver() != TorchLoader()


@torch_available
def test_torch_saver_equal_true() -> None:
    assert TorchSaver().equal(TorchSaver())


@torch_available
def test_torch_saver_equal_false_different_kwargs() -> None:
    assert not TorchSaver(pickle_protocol=5).equal(TorchSaver(pickle_protocol=4))


@torch_available
def test_torch_saver_equal_false_different_type() -> None:
    assert not TorchSaver().equal(TorchLoader())


@torch_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_torch_saver_equal_nan(equal_nan: bool) -> None:
    assert TorchSaver().equal(TorchSaver(), equal_nan=equal_nan)


@torch_available
def test_torch_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    saver = TorchSaver()
    saver.save({"key1": [1, 2, 3], "key2": "abc", "key3": torch.arange(5)}, path)
    assert path.is_file()


@torch_available
def test_torch_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    save_torch({"key1": [1, 2, 3], "key2": "abc", "key3": torch.arange(5)}, path)
    saver = TorchSaver()
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        saver.save({"key1": [1, 2, 3], "key2": "abc", "key3": torch.arange(5)}, path)


@torch_available
def test_torch_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    save_torch({"key1": [1, 2, 3], "key2": "abc"}, path)
    saver = TorchSaver()
    saver.save({"key1": [3, 2, 1], "key2": "meow", "key3": torch.arange(5)}, path, exist_ok=True)
    assert path.is_file()
    assert objects_are_equal(
        load_torch(path), {"key1": [3, 2, 1], "key2": "meow", "key3": torch.arange(5)}
    )


@torch_available
def test_torch_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    path.mkdir(parents=True, exist_ok=True)
    saver = TorchSaver()
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        saver.save({"key1": [1, 2, 3], "key2": "abc", "key3": torch.arange(5)}, path)


def test_torch_saver_no_torch() -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."),
    ):
        TorchSaver()


################################
#     Tests for load_torch     #
################################


@torch_available
def test_load_torch(path_torch: Path) -> None:
    assert objects_are_equal(
        load_torch(path_torch), {"key1": [1, 2, 3], "key2": "abc", "key3": torch.arange(5)}
    )


@torch_available
@torch_greater_equal_1_13
def test_load_torch_weights_only_false(path_torch: Path) -> None:
    assert objects_are_equal(
        load_torch(path_torch, weights_only=False),
        {"key1": [1, 2, 3], "key2": "abc", "key3": torch.arange(5)},
    )


def test_load_torch_no_torch(tmp_path: Path) -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."),
    ):
        load_torch(tmp_path)


################################
#     Tests for save_torch     #
################################


@torch_available
def test_save_torch(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    save_torch({"key1": [1, 2, 3], "key2": "abc", "key3": torch.arange(5)}, path)
    assert path.is_file()


@torch_available
def test_save_torch_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    save_torch({"key1": [1, 2, 3], "key2": "abc", "key3": torch.arange(5)}, path)
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        save_torch({"key1": [1, 2, 3], "key2": "abc", "key3": torch.arange(5)}, path)


@torch_available
def test_save_torch_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    save_torch({"key1": [1, 2, 3], "key2": "abc"}, path)
    save_torch({"key1": [3, 2, 1], "key2": "meow", "key3": torch.arange(5)}, path, exist_ok=True)
    assert path.is_file()
    assert objects_are_equal(
        load_torch(path), {"key1": [3, 2, 1], "key2": "meow", "key3": torch.arange(5)}
    )


@torch_available
def test_save_torch_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        save_torch({"key1": [1, 2, 3], "key2": "abc", "key3": torch.arange(5)}, path)


def test_save_torch_no_torch(tmp_path: Path) -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."),
    ):
        save_torch(
            {"key1": [1, 2, 3], "key2": "abc"},
            tmp_path,
        )


########################################
#     Tests for get_loader_mapping     #
########################################


@torch_available
def test_get_loader_mapping() -> None:
    assert get_loader_mapping() == {"pt": TorchLoader()}


def test_get_loader_mapping_no_torch() -> None:
    with patch("iden.io.torch.is_torch_available", lambda: False):
        assert get_loader_mapping() == {}
