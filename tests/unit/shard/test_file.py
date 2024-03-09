from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iden.constants import KWARGS, LOADER
from iden.io import save_json
from iden.shard import FileShard, create_json_shard
from iden.utils.path import sanitize_path

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path_ = tmp_path_factory.mktemp("tmp").joinpath("data.json")
    save_json([1, 2, 3], path_)
    return path_


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    config = {KWARGS: {"path": sanitize_path(path).as_posix()}, LOADER: ""}
    save_json(config, sanitize_path(uri_))
    return uri_


###############################
#     Tests for FileShard     #
###############################


def test_file_shard_str(uri: str, path: Path) -> None:
    assert str(FileShard(uri=uri, path=path)).startswith("FileShard(")


def test_file_shard_path(uri: str, path: Path) -> None:
    assert FileShard(uri=uri, path=path).path == path


def test_file_shard_equal_true(uri: str, path: Path) -> None:
    assert FileShard(uri=uri, path=path).equal(FileShard(uri=uri, path=path))


def test_file_shard_equal_false_different_uri(uri: str, path: Path) -> None:
    assert not FileShard(uri=uri, path=path).equal(FileShard(uri="", path=path))


def test_file_shard_equal_false_different_path(uri: str, path: Path, tmp_path: Path) -> None:
    assert not FileShard(uri=uri, path=path).equal(FileShard(uri=uri, path=tmp_path))


def test_file_shard_equal_false_different_type(uri: str, path: Path) -> None:
    assert not FileShard(uri=uri, path=path).equal(42)


@pytest.mark.parametrize("equal_nan", [True, False])
def test_file_shard_equal_nan(tmp_path: Path, equal_nan: bool) -> None:
    shard = create_json_shard(
        data={"key1": [1, 2, float("nan")], "key2": "abc"}, uri=tmp_path.joinpath("uri").as_uri()
    )
    assert FileShard.from_uri(uri=shard.get_uri()).equal(
        FileShard.from_uri(uri=shard.get_uri()), equal_nan=equal_nan
    )


def test_file_shard_get_data(uri: str, path: Path) -> None:
    assert FileShard(uri=uri, path=path).get_data() == [1, 2, 3]


def test_file_shard_get_data_multiple_calls(uri: str, path: Path) -> None:
    shard = FileShard(uri=uri, path=path)
    assert shard.get_data() == [1, 2, 3]
    assert shard.get_data() == [1, 2, 3]
    shard.get_data().append(4)
    assert shard.get_data() == [1, 2, 3, 4]


def test_file_shard_get_uri(uri: str, path: Path) -> None:
    assert FileShard(uri=uri, path=path).get_uri() == uri


# TODO(tibo): add later
#  2
# def test_file_shard_from_uri(uri: str, path: Path) -> None:
#     shard = FileShard.from_uri(uri)
#     assert shard.equal(FileShard(uri=uri, path=path))
#     assert shard.get_data() == [1, 2, 3]
