from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from iden.io import YamlLoader, YamlSaver, load_yaml, save_yaml
from iden.testing import yaml_available

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


@yaml_available
def test_yaml_loader_str() -> None:
    assert str(YamlLoader()).startswith("YamlLoader(")


@yaml_available
def test_yaml_loader_equal_true() -> None:
    assert YamlLoader().equal(YamlLoader())


@yaml_available
def test_yaml_loader_equal_false() -> None:
    assert not YamlLoader().equal(YamlSaver())


@yaml_available
def test_yaml_loader_equal_false_child() -> None:
    class Child(YamlLoader): ...

    assert not YamlLoader().equal(Child())


@yaml_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_yaml_loader_equal_nan(equal_nan: bool) -> None:
    assert YamlLoader().equal(YamlLoader(), equal_nan=equal_nan)


@yaml_available
def test_yaml_loader_load(path_yaml: Path) -> None:
    assert YamlLoader().load(path_yaml) == {"key1": [1, 2, 3], "key2": "abc"}


def test_yaml_loader_no_yaml() -> None:
    with (
        patch("iden.utils.imports.yaml.is_yaml_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'yaml' package is required but not installed."),
    ):
        YamlLoader()


###############################
#     Tests for YamlSaver     #
###############################


@yaml_available
def test_yaml_saver_str() -> None:
    assert str(YamlSaver()).startswith("YamlSaver(")


@yaml_available
def test_yaml_saver_equal_true() -> None:
    assert YamlSaver().equal(YamlSaver())


@yaml_available
def test_yaml_saver_equal_false() -> None:
    assert not YamlSaver().equal(YamlLoader())


@yaml_available
def test_yaml_saver_equal_false_child() -> None:
    class Child(YamlSaver): ...

    assert not YamlSaver().equal(Child())


@yaml_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_yaml_saver_equal_nan(equal_nan: bool) -> None:
    assert YamlSaver().equal(YamlSaver(), equal_nan=equal_nan)


@yaml_available
def test_yaml_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    saver = YamlSaver()
    saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


@yaml_available
def test_yaml_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
    saver = YamlSaver()
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


@yaml_available
def test_yaml_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
    saver = YamlSaver()
    saver.save({"key1": [3, 2, 1], "key2": "meow"}, path, exist_ok=True)
    assert path.is_file()
    assert load_yaml(path) == {"key1": [3, 2, 1], "key2": "meow"}


@yaml_available
def test_yaml_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    path.mkdir(parents=True, exist_ok=True)
    saver = YamlSaver()
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


def test_yaml_saver_no_yaml() -> None:
    with (
        patch("iden.utils.imports.yaml.is_yaml_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'yaml' package is required but not installed."),
    ):
        YamlSaver()


###############################
#     Tests for load_yaml     #
###############################


@yaml_available
def test_load_yaml(path_yaml: Path) -> None:
    assert load_yaml(path_yaml) == {"key1": [1, 2, 3], "key2": "abc"}


###############################
#     Tests for save_yaml     #
###############################


@yaml_available
def test_save_yaml(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


@yaml_available
def test_save_yaml_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)


@yaml_available
def test_save_yaml_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
    save_yaml({"key1": [3, 2, 1], "key2": "meow"}, path, exist_ok=True)
    assert path.is_file()
    assert load_yaml(path) == {"key1": [3, 2, 1], "key2": "meow"}


@yaml_available
def test_save_yaml_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.yaml")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        save_yaml({"key1": [1, 2, 3], "key2": "abc"}, path)
