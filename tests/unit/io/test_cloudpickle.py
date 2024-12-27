from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iden.io import (
    CloudpickleLoader,
    CloudpickleSaver,
    load_cloudpickle,
    save_cloudpickle,
    save_text,
)
from iden.io.cloudpickle import get_loader_mapping
from iden.testing import cloudpickle_available

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_pickle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.pkl")
    CloudpickleSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
    return path


#######################################
#     Tests for CloudpickleLoader     #
#######################################


@cloudpickle_available
def test_cloudpickle_loader_repr() -> None:
    assert repr(CloudpickleLoader()).startswith("CloudpickleLoader(")


@cloudpickle_available
def test_cloudpickle_loader_str() -> None:
    assert str(CloudpickleLoader()).startswith("CloudpickleLoader(")


@cloudpickle_available
def test_cloudpickle_loader_eq_true() -> None:
    assert CloudpickleLoader() == CloudpickleLoader()


@cloudpickle_available
def test_cloudpickle_loader_eq_false() -> None:
    assert CloudpickleLoader() != CloudpickleSaver()


@cloudpickle_available
def test_cloudpickle_loader_load(path_pickle: Path) -> None:
    assert CloudpickleLoader().load(path_pickle) == {"key1": [1, 2, 3], "key2": "abc"}


######################################
#     Tests for CloudpickleSaver     #
######################################


@cloudpickle_available
def test_cloudpickle_saver_repr() -> None:
    assert repr(CloudpickleSaver()) == "CloudpickleSaver()"


@cloudpickle_available
def test_cloudpickle_saver_str() -> None:
    assert str(CloudpickleSaver()) == "CloudpickleSaver()"


@cloudpickle_available
def test_cloudpickle_saver_eq_true() -> None:
    assert CloudpickleSaver() == CloudpickleSaver()


@cloudpickle_available
def test_cloudpickle_saver_eq_false() -> None:
    assert CloudpickleSaver() != CloudpickleLoader()


@cloudpickle_available
def test_cloudpickle_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    saver = CloudpickleSaver()
    saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


@cloudpickle_available
def test_cloudpickle_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_text("hello", path)
    saver = CloudpickleSaver()
    with pytest.raises(FileExistsError, match="path .* already exists."):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


@cloudpickle_available
def test_cloudpickle_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_text("hello", path)
    saver = CloudpickleSaver()
    saver.save({"key1": [3, 2, 1], "key2": "abc"}, path, exist_ok=True)
    assert path.is_file()
    assert load_cloudpickle(path) == {"key1": [3, 2, 1], "key2": "abc"}


@cloudpickle_available
def test_cloudpickle_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    path.mkdir(parents=True, exist_ok=True)
    saver = CloudpickleSaver()
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


######################################
#     Tests for load_cloudpickle     #
######################################


@cloudpickle_available
def test_load_cloudpickle(path_pickle: Path) -> None:
    assert load_cloudpickle(path_pickle) == {"key1": [1, 2, 3], "key2": "abc"}


######################################
#     Tests for save_cloudpickle     #
######################################


@cloudpickle_available
def test_save_cloudpickle(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_cloudpickle({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


@cloudpickle_available
def test_save_cloudpickle_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_text("hello", path)
    with pytest.raises(FileExistsError, match="path .* already exists."):
        save_cloudpickle({"key1": [1, 2, 3], "key2": "abc"}, path)


@cloudpickle_available
def test_save_cloudpickle_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_text("hello", path)
    save_cloudpickle({"key1": [3, 2, 1], "key2": "abc"}, path, exist_ok=True)
    assert path.is_file()
    assert load_cloudpickle(path) == {"key1": [3, 2, 1], "key2": "abc"}


@cloudpickle_available
def test_save_cloudpickle_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        save_cloudpickle({"key1": [1, 2, 3], "key2": "abc"}, path)


########################################
#     Tests for get_loader_mapping     #
########################################


@cloudpickle_available
def test_get_loader_mapping() -> None:
    assert get_loader_mapping() == {}
