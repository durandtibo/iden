from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iden.io import TextLoader, TextSaver, load_text, save_text
from iden.io.text import get_loader_mapping

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_text(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.txt")
    save_text("hello", path)
    return path


################################
#     Tests for TextLoader     #
################################


def test_text_loader_str() -> None:
    assert str(TextLoader()).startswith("TextLoader(")


def test_text_loader_eq_true() -> None:
    assert TextLoader() == TextLoader()


def test_text_loader_eq_false() -> None:
    assert TextLoader() != TextSaver()


def test_text_loader_equal_true() -> None:
    assert TextLoader().equal(TextLoader())


def test_text_loader_equal_false() -> None:
    assert not TextLoader().equal(TextSaver())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_text_loader_equal_nan(equal_nan: bool) -> None:
    assert TextLoader().equal(TextLoader(), equal_nan=equal_nan)


def test_text_loader_load(path_text: Path) -> None:
    assert TextLoader().load(path_text) == "hello"


###############################
#     Tests for TextSaver     #
###############################


def test_text_saver_str() -> None:
    assert str(TextSaver()).startswith("TextSaver(")


def test_text_saver_eq_true() -> None:
    assert TextSaver() == TextSaver()


def test_text_saver_eq_false() -> None:
    assert TextSaver() != TextLoader()


def test_text_saver_equal_true() -> None:
    assert TextSaver().equal(TextSaver())


def test_text_saver_equal_false() -> None:
    assert not TextSaver().equal(TextLoader())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_text_saver_equal_nan(equal_nan: bool) -> None:
    assert TextSaver().equal(TextSaver(), equal_nan=equal_nan)


def test_text_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    saver = TextSaver()
    saver.save("hello", path)
    assert path.is_file()


def test_text_saver_save_list(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    saver = TextSaver()
    saver.save([1, 2, 3], path)
    assert path.is_file()
    assert load_text(path) == "[1, 2, 3]"


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


###############################
#     Tests for load_text     #
###############################


def test_load_text(path_text: Path) -> None:
    assert load_text(path_text) == "hello"


###############################
#     Tests for save_text     #
###############################


def test_save_text(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    assert path.is_file()


def test_save_text_list(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text([1, 2, 3], path)
    assert path.is_file()
    assert load_text(path) == "[1, 2, 3]"


def test_save_text_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    with pytest.raises(FileExistsError, match="path .* already exists."):
        save_text("hello", path)


def test_save_text_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    save_text("meow", path, exist_ok=True)
    assert path.is_file()
    assert load_text(path) == "meow"


def test_save_text_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        save_text("hello", path)


########################################
#     Tests for get_loader_mapping     #
########################################


def test_get_loader_mapping() -> None:
    assert get_loader_mapping() == {"txt": TextLoader()}
