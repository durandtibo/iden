from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iden.io import TextLoader, TextSaver
from iden.utils.io import load_text, save_text

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_text(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.txt")
    TextSaver().save("hello", path)
    return path


################################
#     Tests for TextLoader     #
################################


def test_text_loader_str() -> None:
    assert str(TextLoader()).startswith("TextLoader(")


def test_text_loader_load(path_text: Path) -> None:
    assert TextLoader().load(path_text) == "hello"


###############################
#     Tests for TextSaver     #
###############################


def test_text_saver_str() -> None:
    assert str(TextSaver()).startswith("TextSaver(")


def test_text_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    saver = TextSaver()
    saver.save("hello", path)
    assert path.is_file()


def test_text_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    saver = TextSaver()
    with pytest.raises(FileExistsError, match="path .* already exists."):
        saver.save("hello", path)


def test_text_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    saver = TextSaver()
    saver.save("meow", path, exist_ok=True)
    assert path.is_file()
    assert load_text(path) == "meow"


def test_text_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    path.mkdir(parents=True, exist_ok=True)
    saver = TextSaver()
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        saver.save("hello", path)
