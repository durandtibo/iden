from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iden.io import YamlLoader, YamlSaver, load_yaml, save_yaml
from iden.io.yaml import get_loader_mapping

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_yaml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.yaml")
    save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
    return path


################################
#     Tests for YamlLoader     #
################################


def test_yaml_loader_str() -> None:
    assert str(YamlLoader()).startswith("YamlLoader(")


def test_yaml_loader_eq_true() -> None:
    assert YamlLoader() == YamlLoader()


def test_yaml_loader_eq_false() -> None:
    assert YamlLoader() != YamlSaver()


def test_yaml_loader_load(path_yaml: Path) -> None:
    assert YamlLoader().load(path_yaml) == {"key1": [1, 2, 3], "key2": "abc"}


###############################
#     Tests for YamlSaver     #
###############################


def test_yaml_saver_str() -> None:
    assert str(YamlSaver()).startswith("YamlSaver(")


def test_yaml_saver_eq_true() -> None:
    assert YamlSaver() == YamlSaver()


def test_yaml_saver_eq_false() -> None:
    assert YamlSaver() != YamlLoader()


def test_yaml_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    saver = YamlSaver()
    saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


def test_yaml_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
    saver = YamlSaver()
    with pytest.raises(FileExistsError, match="path .* already exists."):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


def test_yaml_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
    saver = YamlSaver()
    saver.save({"key1": [3, 2, 1], "key2": "meow"}, path, exist_ok=True)
    assert path.is_file()
    assert load_yaml(path) == {"key1": [3, 2, 1], "key2": "meow"}


def test_yaml_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    path.mkdir(parents=True, exist_ok=True)
    saver = YamlSaver()
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


###############################
#     Tests for load_yaml     #
###############################


def test_load_yaml(path_yaml: Path) -> None:
    assert load_yaml(path_yaml) == {"key1": [1, 2, 3], "key2": "abc"}


###############################
#     Tests for save_yaml     #
###############################


def test_save_yaml(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


def test_save_yaml_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
    with pytest.raises(FileExistsError, match="path .* already exists."):
        save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)


def test_save_yaml_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
    save_yaml({"key1": [3, 2, 1], "key2": "meow"}, path, exist_ok=True)
    assert path.is_file()
    assert load_yaml(path) == {"key1": [3, 2, 1], "key2": "meow"}


def test_save_yaml_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match="path .* is a directory"):
        save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)


########################################
#     Tests for get_loader_mapping     #
########################################


def test_get_loader_mapping() -> None:
    assert get_loader_mapping() == {"yaml": YamlLoader(), "yml": YamlLoader()}
