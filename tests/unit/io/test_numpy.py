from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from coola import objects_are_equal
from coola.testing import numpy_available
from coola.utils import is_numpy_available

from iden.io import NumpyZLoader, NumpyZSaver

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def data() -> dict[str, np.ndarray]:
    return {"key1": np.array([1, 2, 3]), "key2": np.ones((2, 3)), "key3": np.arange(5)}


@pytest.fixture(scope="module")
def path_numpy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.npz")
    NumpyZSaver().save(
        {"key1": np.array([1, 2, 3]), "key2": np.ones((2, 3)), "key3": np.arange(5)}, path
    )
    return path


##################################
#     Tests for NumpyZLoader     #
##################################


@numpy_available
def test_numpyz_loader_str() -> None:
    assert str(NumpyZLoader()).startswith("NumpyZLoader(")


@numpy_available
def test_numpyz_loader_eq_true() -> None:
    assert NumpyZLoader() == NumpyZLoader()


@numpy_available
def test_numpyz_loader_eq_false() -> None:
    assert NumpyZLoader() != NumpyZSaver()


@numpy_available
def test_numpyz_loader_load_dict(path_numpy: Path) -> None:
    assert objects_are_equal(
        NumpyZLoader().load(path_numpy),
        {
            "key1": np.array([1, 2, 3]),
            "key2": np.ones((2, 3)),
            "key3": np.arange(5),
        },
    )


@numpy_available
def test_numpyz_loader_load_seq(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.npz")
    NumpyZSaver().save([np.array([1, 2, 3]), np.ones((2, 3)), np.arange(5)], path)
    assert objects_are_equal(
        NumpyZLoader().load(path),
        {
            "arr_0": np.array([1, 2, 3]),
            "arr_1": np.ones((2, 3)),
            "arr_2": np.arange(5),
        },
    )


def test_numpyz_loader_no_numpy() -> None:
    with (
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match="`numpy` package is required but not installed."),
    ):
        NumpyZLoader()


#################################
#     Tests for NumpyZSaver     #
#################################


@numpy_available
def test_numpyz_saver_str() -> None:
    assert str(NumpyZSaver()).startswith("NumpyZSaver(")


@numpy_available
def test_numpyz_saver_eq_true() -> None:
    assert NumpyZSaver() == NumpyZSaver()


@numpy_available
def test_numpyz_saver_eq_false() -> None:
    assert NumpyZSaver() != NumpyZLoader()


@numpy_available
def test_numpyz_saver_save(tmp_path: Path, data: dict[str, np.ndarray]) -> None:
    path = tmp_path.joinpath("tmp/data.npz")
    saver = NumpyZSaver()
    saver.save(data, path)
    assert path.is_file()


@numpy_available
def test_numpyz_saver_save_file_exist(tmp_path: Path, data: dict[str, np.ndarray]) -> None:
    path = tmp_path.joinpath("tmp/data.npz")
    saver = NumpyZSaver()
    saver.save(data, path)
    with pytest.raises(FileExistsError, match="path .* already exists."):
        saver.save(data, path)


@numpy_available
def test_numpyz_saver_save_file_exist_ok(tmp_path: Path, data: dict[str, np.ndarray]) -> None:
    path = tmp_path.joinpath("tmp/data.npz")
    saver = NumpyZSaver()
    saver.save(data, path)
    saver.save({"key1": np.array([1, 2, 3]), "key2": np.ones((2, 3))}, path, exist_ok=True)
    assert path.is_file()
    assert objects_are_equal(
        NumpyZLoader().load(path), {"key1": np.array([1, 2, 3]), "key2": np.ones((2, 3))}
    )


@numpy_available
def test_numpyz_saver_save_file_exist_ok_dir(tmp_path: Path, data: dict[str, np.ndarray]) -> None:
    path = tmp_path.joinpath("tmp/data.npz")
    path.mkdir(parents=True, exist_ok=True)
    saver = NumpyZSaver()
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        saver.save(data, path)


def test_numpyz_saver_no_numpy() -> None:
    with (
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match="`numpy` package is required but not installed."),
    ):
        NumpyZSaver()
