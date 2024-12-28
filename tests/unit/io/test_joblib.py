from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from iden.io import JoblibLoader, JoblibSaver, load_joblib, save_joblib, save_text
from iden.io.joblib import get_loader_mapping
from iden.testing import joblib_available

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_joblib(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.joblib")
    JoblibSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
    return path


##################################
#     Tests for JoblibLoader     #
##################################


@joblib_available
def test_joblib_loader_repr() -> None:
    assert repr(JoblibLoader()) == "JoblibLoader()"


@joblib_available
def test_joblib_loader_str() -> None:
    assert str(JoblibLoader()) == "JoblibLoader()"


@joblib_available
def test_joblib_loader_eq_true() -> None:
    assert JoblibLoader() == JoblibLoader()


@joblib_available
def test_joblib_loader_eq_false() -> None:
    assert JoblibLoader() != JoblibSaver()


@joblib_available
def test_joblib_loader_load(path_joblib: Path) -> None:
    assert JoblibLoader().load(path_joblib) == {"key1": [1, 2, 3], "key2": "abc"}


#################################
#     Tests for JoblibSaver     #
#################################


@joblib_available
def test_joblib_saver_repr() -> None:
    assert repr(JoblibSaver()) == "JoblibSaver()"


@joblib_available
def test_joblib_saver_repr_with_kwargs() -> None:
    assert repr(JoblibSaver(compress=3)) == "JoblibSaver(compress=3)"


@joblib_available
def test_joblib_saver_str() -> None:
    assert str(JoblibSaver()) == "JoblibSaver()"


@joblib_available
def test_joblib_saver_str_with_kwargs() -> None:
    assert str(JoblibSaver(compress=3)) == "JoblibSaver(compress=3)"


@joblib_available
def test_joblib_saver_eq_true() -> None:
    assert JoblibSaver() == JoblibSaver()


@joblib_available
def test_joblib_saver_eq_false_different_kwargs() -> None:
    assert JoblibSaver(compress=3) != JoblibSaver(compress=2)


@joblib_available
def test_joblib_saver_eq_false_different_type() -> None:
    assert JoblibSaver() != JoblibLoader()


@joblib_available
def test_joblib_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.joblib")
    saver = JoblibSaver()
    saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


@joblib_available
def test_joblib_saver_save_compress_3(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.joblib")
    saver = JoblibSaver(compress=3)
    saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


@joblib_available
def test_joblib_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.joblib")
    save_text("hello", path)
    saver = JoblibSaver()
    with pytest.raises(FileExistsError, match="path .* already exists."):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


@joblib_available
def test_joblib_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.joblib")
    save_text("hello", path)
    saver = JoblibSaver()
    saver.save({"key1": [3, 2, 1], "key2": "abc"}, path, exist_ok=True)
    assert path.is_file()
    assert load_joblib(path) == {"key1": [3, 2, 1], "key2": "abc"}


@joblib_available
def test_joblib_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.joblib")
    path.mkdir(parents=True, exist_ok=True)
    saver = JoblibSaver()
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


#################################
#     Tests for load_joblib     #
#################################


@joblib_available
def test_load_joblib(path_joblib: Path) -> None:
    assert load_joblib(path_joblib) == {"key1": [1, 2, 3], "key2": "abc"}


#################################
#     Tests for save_joblib     #
#################################


@joblib_available
def test_save_joblib(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.joblib")
    save_joblib({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


@joblib_available
def test_save_joblib_compress_3(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.joblib")
    save_joblib({"key1": [1, 2, 3], "key2": "abc"}, path, compress=3)
    assert path.is_file()


@joblib_available
def test_save_joblib_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.joblib")
    save_text("hello", path)
    with pytest.raises(FileExistsError, match="path .* already exists."):
        save_joblib({"key1": [1, 2, 3], "key2": "abc"}, path)


@joblib_available
def test_save_joblib_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.joblib")
    save_text("hello", path)
    save_joblib({"key1": [3, 2, 1], "key2": "abc"}, path, exist_ok=True)
    assert path.is_file()
    assert load_joblib(path) == {"key1": [3, 2, 1], "key2": "abc"}


@joblib_available
def test_save_joblib_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.joblib")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        save_joblib({"key1": [1, 2, 3], "key2": "abc"}, path)


########################################
#     Tests for get_loader_mapping     #
########################################


@joblib_available
def test_get_loader_mapping() -> None:
    assert get_loader_mapping() == {"joblib": JoblibLoader()}


def test_get_loader_mapping_no_joblib() -> None:
    with patch("iden.io.joblib.is_joblib_available", lambda: False):
        assert get_loader_mapping() == {}
