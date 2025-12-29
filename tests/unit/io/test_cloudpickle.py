from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from iden.io import (
    CloudpickleLoader,
    CloudpickleSaver,
    load_cloudpickle,
    save_cloudpickle,
    save_text,
)
from iden.testing import cloudpickle_available
from iden.utils.imports import is_cloudpickle_available

if is_cloudpickle_available():
    from cloudpickle import DEFAULT_PROTOCOL
else:
    DEFAULT_PROTOCOL = 1

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
    assert repr(CloudpickleLoader()) == "CloudpickleLoader()"


@cloudpickle_available
def test_cloudpickle_loader_str() -> None:
    assert str(CloudpickleLoader()) == "CloudpickleLoader()"


@cloudpickle_available
def test_cloudpickle_loader_equal_true() -> None:
    assert CloudpickleLoader().equal(CloudpickleLoader())


@cloudpickle_available
def test_cloudpickle_loader_equal_false_different_type() -> None:
    assert not CloudpickleLoader().equal(CloudpickleSaver())


@cloudpickle_available
def test_cloudpickle_loader_equal_false_different_type_child() -> None:
    class Child(CloudpickleLoader): ...

    assert not CloudpickleLoader().equal(Child())


@cloudpickle_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_cloudpickle_loader_equal_nan(equal_nan: bool) -> None:
    assert CloudpickleLoader().equal(CloudpickleLoader(), equal_nan=equal_nan)


@cloudpickle_available
def test_cloudpickle_loader_load(path_pickle: Path) -> None:
    assert CloudpickleLoader().load(path_pickle) == {"key1": [1, 2, 3], "key2": "abc"}


def test_cloudpickle_loader_no_cloudpickle() -> None:
    with (
        patch("iden.utils.imports.is_cloudpickle_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'cloudpickle' package is required but not installed."),
    ):
        CloudpickleLoader()


######################################
#     Tests for CloudpickleSaver     #
######################################


@cloudpickle_available
def test_cloudpickle_saver_repr() -> None:
    assert repr(CloudpickleSaver()) == "CloudpickleSaver()"


@cloudpickle_available
def test_cloudpickle_saver_repr_with_kwargs() -> None:
    assert repr(CloudpickleSaver(protocol=5)) == "CloudpickleSaver(protocol=5)"


@cloudpickle_available
def test_cloudpickle_saver_str() -> None:
    assert str(CloudpickleSaver()) == "CloudpickleSaver()"


@cloudpickle_available
def test_cloudpickle_saver_str_with_kwargs() -> None:
    assert str(CloudpickleSaver(protocol=5)) == "CloudpickleSaver(protocol=5)"


@cloudpickle_available
def test_cloudpickle_saver_equal_true() -> None:
    assert CloudpickleSaver().equal(CloudpickleSaver())


@cloudpickle_available
def test_cloudpickle_saver_equal_false_different_kwargs() -> None:
    assert not CloudpickleSaver(protocol=5).equal(CloudpickleSaver(protocol=4))


@cloudpickle_available
def test_cloudpickle_saver_equal_false_different_type_child() -> None:
    assert not CloudpickleSaver().equal(CloudpickleLoader())


@cloudpickle_available
def test_cloudpickle_saver_equal_false_different_type() -> None:
    class Child(CloudpickleSaver): ...

    assert not CloudpickleSaver().equal(Child())


@cloudpickle_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_cloudpickle_saver_equal_nan(equal_nan: bool) -> None:
    assert CloudpickleSaver().equal(CloudpickleSaver(), equal_nan=equal_nan)


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
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
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
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


@cloudpickle_available
@pytest.mark.parametrize("protocol", list(range(1, DEFAULT_PROTOCOL + 1)))
def test_cloudpickle_saver_save_protocol(tmp_path: Path, protocol: int) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    saver = CloudpickleSaver(protocol=protocol)
    saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


def test_cloudpickle_saver_no_cloudpickle() -> None:
    with (
        patch("iden.utils.imports.is_cloudpickle_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'cloudpickle' package is required but not installed."),
    ):
        CloudpickleSaver()


######################################
#     Tests for load_cloudpickle     #
######################################


@cloudpickle_available
def test_load_cloudpickle(path_pickle: Path) -> None:
    assert load_cloudpickle(path_pickle) == {"key1": [1, 2, 3], "key2": "abc"}


def test_load_cloudpickle_no_cloudpickle(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_cloudpickle_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'cloudpickle' package is required but not installed."),
    ):
        load_cloudpickle(tmp_path.joinpath("data.pkl"))


######################################
#     Tests for save_cloudpickle     #
######################################


@cloudpickle_available
def test_save_cloudpickle(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_cloudpickle({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


@cloudpickle_available
@pytest.mark.parametrize("protocol", list(range(1, DEFAULT_PROTOCOL + 1)))
def test_save_cloudpickle_protocol(tmp_path: Path, protocol: int) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_cloudpickle({"key1": [1, 2, 3], "key2": "abc"}, path, protocol=protocol)
    assert path.is_file()


@cloudpickle_available
def test_save_cloudpickle_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_text("hello", path)
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
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
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        save_cloudpickle({"key1": [1, 2, 3], "key2": "abc"}, path)


def test_save_cloudpickle_no_cloudpickle(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_cloudpickle_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'cloudpickle' package is required but not installed."),
    ):
        save_cloudpickle({"key1": [1, 2, 3], "key2": "abc"}, tmp_path.joinpath("data.pkl"))
