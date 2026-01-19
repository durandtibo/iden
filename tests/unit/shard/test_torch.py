from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from coola.equality import objects_are_equal
from coola.testing.fixtures import torch_available
from coola.utils.imports import is_torch_available
from objectory import OBJECT_TARGET

from iden.constants import KWARGS, LOADER
from iden.io import load_json
from iden.shard import TorchShard, create_torch_shard

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.pt")


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_torch_shard(
        data={"key1": torch.ones(2, 3), "key2": torch.arange(5)}, uri=uri_, path=path
    )
    return uri_


################################
#     Tests for TorchShard     #
################################


@torch_available
def test_torch_shard_repr(uri: str, path: Path) -> None:
    assert repr(TorchShard(uri=uri, path=path)).startswith("TorchShard(")


@torch_available
def test_torch_shard_str(uri: str, path: Path) -> None:
    assert str(TorchShard(uri=uri, path=path)).startswith("TorchShard(")


@torch_available
def test_torch_shard_path(uri: str, path: Path) -> None:
    assert TorchShard(uri=uri, path=path).path == path


@torch_available
def test_torch_shard_clear_not_initialized(uri: str, path: Path) -> None:
    shard = TorchShard(uri=uri, path=path)
    shard.clear()
    assert not shard._is_cached
    assert shard._data is None


@torch_available
def test_torch_shard_clear_is_cached(uri: str, path: Path) -> None:
    shard = TorchShard(uri=uri, path=path)
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": torch.ones(2, 3), "key2": torch.arange(5)}
    )
    assert shard.is_cached()
    shard.clear()
    assert not shard.is_cached()


@torch_available
def test_torch_shard_equal_true(uri: str, path: Path) -> None:
    assert TorchShard(uri=uri, path=path).equal(TorchShard(uri=uri, path=path))


@torch_available
def test_torch_shard_equal_false_different_uri(uri: str, path: Path) -> None:
    assert not TorchShard(uri=uri, path=path).equal(TorchShard(uri="", path=path))


@torch_available
def test_torch_shard_equal_false_different_path(uri: str, path: Path, tmp_path: Path) -> None:
    assert not TorchShard(uri=uri, path=path).equal(TorchShard(uri=uri, path=tmp_path))


@torch_available
def test_torch_shard_equal_false_different_type(uri: str, path: Path) -> None:
    assert not TorchShard(uri=uri, path=path).equal(42)


@torch_available
def test_torch_shard_equal_false_different_type_child(uri: str, path: Path) -> None:
    class Child(TorchShard): ...

    assert not TorchShard(uri=uri, path=path).equal(Child(uri=uri, path=path))


@torch_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_torch_shard_equal_nan(tmp_path: Path, equal_nan: bool) -> None:
    shard = create_torch_shard(
        data={"key1": [1, 2, float("nan")], "key2": "abc"}, uri=tmp_path.joinpath("uri").as_uri()
    )
    assert shard.equal(TorchShard.from_uri(uri=shard.get_uri()), equal_nan=equal_nan)


@torch_available
def test_torch_shard_get_data(uri: str, path: Path) -> None:
    assert objects_are_equal(
        TorchShard(uri=uri, path=path).get_data(),
        {"key1": torch.ones(2, 3), "key2": torch.arange(5)},
    )


@torch_available
def test_torch_shard_get_data_cache_false_not_cached(uri: str, path: Path) -> None:
    shard = TorchShard(uri=uri, path=path)
    assert not shard.is_cached()
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})
    assert not shard.is_cached()


@torch_available
def test_torch_shard_get_data_cache_false_cached(uri: str, path: Path) -> None:
    shard = TorchShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})
    assert shard.is_cached()


@torch_available
def test_torch_shard_get_data_cache_true_not_cached(uri: str, path: Path) -> None:
    shard = TorchShard(uri=uri, path=path)
    assert not shard.is_cached()
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": torch.ones(2, 3), "key2": torch.arange(5)}
    )
    assert shard.is_cached()


@torch_available
def test_torch_shard_get_data_cache_true_cached(uri: str, path: Path) -> None:
    shard = TorchShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": torch.ones(2, 3), "key2": torch.arange(5)}
    )
    assert shard.is_cached()


@torch_available
def test_torch_shard_get_data_multiple_calls(uri: str, path: Path) -> None:
    shard = TorchShard(uri=uri, path=path)
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": torch.ones(2, 3), "key2": torch.arange(5)}
    )
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})
    shard.get_data()["key1"].add_(1.0)
    assert objects_are_equal(
        shard.get_data(), {"key1": torch.full((2, 3), 2.0), "key2": torch.arange(5)}
    )


@torch_available
def test_torch_shard_get_uri(uri: str, path: Path) -> None:
    assert TorchShard(uri=uri, path=path).get_uri() == uri


@torch_available
def test_torch_shard_is_cached_false(uri: str, path: Path) -> None:
    shard = TorchShard(uri=uri, path=path)
    assert not shard.is_cached()


@torch_available
def test_torch_shard_is_cached_true(uri: str, path: Path) -> None:
    shard = TorchShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()


@torch_available
def test_torch_shard_from_uri(uri: str, path: Path) -> None:
    shard = TorchShard.from_uri(uri)
    assert shard.equal(TorchShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


def test_torch_shard_generate_uri_config(path: Path) -> None:
    assert TorchShard.generate_uri_config(path) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.TorchShardLoader"},
    }


def test_torch_shard_no_torch(tmp_path: Path) -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."),
    ):
        TorchShard(uri="", path=tmp_path)


########################################
#     Tests for create_torch_shard     #
########################################


@torch_available
def test_create_torch_shard(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("my_uri.pt")
    shard = create_torch_shard(data={"key1": torch.ones(2, 3), "key2": torch.arange(5)}, uri=uri)

    assert uri_file.is_file()
    assert load_json(uri_file) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.TorchShardLoader"},
    }
    assert shard.equal(TorchShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


@torch_available
def test_create_torch_shard_with_data(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("data.pt")
    shard = create_torch_shard(
        data={"key1": torch.ones(2, 3), "key2": torch.arange(5)}, uri=uri, path=path
    )

    assert load_json(uri_file) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.TorchShardLoader"},
    }
    assert shard.equal(TorchShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})
