from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iden.io import PickleLoader, PickleSaver
from iden.utils.io import load_pickle, save_text

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_pickle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.pkl")
    PickleSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
    return path


##################################
#     Tests for PickleLoader     #
##################################


def test_pickle_loader_str() -> None:
    assert str(PickleLoader()).startswith("PickleLoader(")


def test_pickle_loader_load(path_pickle: Path) -> None:
    assert PickleLoader().load(path_pickle) == {"key1": [1, 2, 3], "key2": "abc"}


#################################
#     Tests for PickleSaver     #
#################################


def test_pickle_saver_str() -> None:
    assert str(PickleSaver()).startswith("PickleSaver(")


def test_pickle_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    saver = PickleSaver()
    saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


def test_pickle_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_text("hello", path)
    saver = PickleSaver()
    with pytest.raises(FileExistsError, match="path .* already exists."):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


def test_pickle_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_text("hello", path)
    saver = PickleSaver()
    saver.save({"key1": [3, 2, 1], "key2": "abc"}, path, exist_ok=True)
    assert path.is_file()
    assert load_pickle(path) == {"key1": [3, 2, 1], "key2": "abc"}


def test_pickle_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    path.mkdir(parents=True, exist_ok=True)
    saver = PickleSaver()
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)
