from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from coola import objects_are_equal
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available
from objectory import OBJECT_TARGET

from iden.constants import KWARGS, LOADER
from iden.io import load_json
from iden.shard import (
    NumpySafetensorsShard,
    TorchSafetensorsShard,
    create_numpy_safetensors_shard,
    create_torch_safetensors_shard,
)
from iden.testing import safetensors_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.safetensors")


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, path: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_torch_safetensors_shard(
        data={"key1": torch.ones(2, 3), "key2": torch.arange(5)}, uri=uri_, path=path
    )
    return uri_


@pytest.fixture(scope="module")
def path_np(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tmp").joinpath("data.safetensors")


@pytest.fixture(scope="module")
def uri_np(tmp_path_factory: pytest.TempPathFactory, path_np: Path) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_numpy_safetensors_shard(
        data={"key1": np.ones((2, 3)), "key2": np.arange(5)}, uri=uri_, path=path_np
    )
    return uri_


###########################################
#     Tests for NumpySafetensorsShard     #
###########################################


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_str(uri_np: str, path_np: Path) -> None:
    assert str(NumpySafetensorsShard(uri=uri_np, path=path_np)).startswith("NumpySafetensorsShard(")


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_path(uri_np: str, path_np: Path) -> None:
    assert NumpySafetensorsShard(uri=uri_np, path=path_np).path == path_np


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_clear_not_initialized(uri_np: str, path_np: Path) -> None:
    shard = NumpySafetensorsShard(uri=uri_np, path=path_np)
    shard.clear()
    assert not shard.is_cached()


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_clear_is_cached(uri_np: str, path_np: Path) -> None:
    shard = NumpySafetensorsShard(uri=uri_np, path=path_np)
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": np.ones((2, 3)), "key2": np.arange(5)}
    )
    assert shard.is_cached()
    shard.clear()
    assert not shard.is_cached()


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_equal_true(uri_np: str, path_np: Path) -> None:
    assert NumpySafetensorsShard(uri=uri_np, path=path_np).equal(
        NumpySafetensorsShard(uri=uri_np, path=path_np)
    )


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_equal_false_different_uri(uri_np: str, path_np: Path) -> None:
    assert not NumpySafetensorsShard(uri=uri_np, path=path_np).equal(
        NumpySafetensorsShard(uri="", path=path_np)
    )


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_equal_false_different_path(
    uri_np: str, path_np: Path, tmp_path: Path
) -> None:
    assert not NumpySafetensorsShard(uri=uri_np, path=path_np).equal(
        NumpySafetensorsShard(uri=uri_np, path=tmp_path)
    )


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_equal_false_different_type(uri_np: str, path_np: Path) -> None:
    assert not NumpySafetensorsShard(uri=uri_np, path=path_np).equal(42)


@safetensors_available
@numpy_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_numpy_safetensors_shard_equal_nan(tmp_path: Path, equal_nan: bool) -> None:
    shard = create_numpy_safetensors_shard(
        data={"key1": np.ones((2, 3)), "key2": np.array([1, 2, float("nan")])},
        uri=tmp_path.joinpath("uri").as_uri(),
    )
    assert shard.equal(NumpySafetensorsShard.from_uri(uri=shard.get_uri()), equal_nan=equal_nan)


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_get_data(uri_np: str, path_np: Path) -> None:
    assert objects_are_equal(
        NumpySafetensorsShard(uri=uri_np, path=path_np).get_data(),
        {
            "key1": np.ones((2, 3)),
            "key2": np.arange(5),
        },
    )


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_get_data_cache_false_not_cached(
    uri_np: str, path_np: Path
) -> None:
    shard = NumpySafetensorsShard(uri=uri_np, path=path_np)
    assert not shard.is_cached()
    assert objects_are_equal(shard.get_data(), {"key1": np.ones((2, 3)), "key2": np.arange(5)})
    assert not shard.is_cached()


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_get_data_cache_false_cached(uri_np: str, path_np: Path) -> None:
    shard = NumpySafetensorsShard(uri=uri_np, path=path_np)
    shard.get_data(cache=True)
    assert shard.is_cached()
    assert objects_are_equal(shard.get_data(), {"key1": np.ones((2, 3)), "key2": np.arange(5)})
    assert shard.is_cached()


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_get_data_cache_true_not_cached(uri_np: str, path_np: Path) -> None:
    shard = NumpySafetensorsShard(uri=uri_np, path=path_np)
    assert not shard.is_cached()
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": np.ones((2, 3)), "key2": np.arange(5)}
    )
    assert shard.is_cached()


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_get_data_cache_true_cached(uri_np: str, path_np: Path) -> None:
    shard = NumpySafetensorsShard(uri=uri_np, path=path_np)
    shard.get_data(cache=True)
    assert shard.is_cached()
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": np.ones((2, 3)), "key2": np.arange(5)}
    )
    assert shard.is_cached()


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_get_data_multiple_calls(uri_np: str, path_np: Path) -> None:
    shard = NumpySafetensorsShard(uri=uri_np, path=path_np)
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": np.ones((2, 3)), "key2": np.arange(5)}
    )
    assert objects_are_equal(shard.get_data(), {"key1": np.ones((2, 3)), "key2": np.arange(5)})
    shard.get_data()["key1"] += 1
    assert objects_are_equal(shard.get_data(), {"key1": np.full((2, 3), 2.0), "key2": np.arange(5)})


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_get_uri(uri_np: str, path_np: Path) -> None:
    assert NumpySafetensorsShard(uri=uri_np, path=path_np).get_uri() == uri_np


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_is_cached_false(uri_np: str, path_np: Path) -> None:
    shard = NumpySafetensorsShard(uri=uri_np, path=path_np)
    assert not shard.is_cached()


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_is_cached_true(uri_np: str, path_np: Path) -> None:
    shard = NumpySafetensorsShard(uri=uri_np, path=path_np)
    shard.get_data(cache=True)
    assert shard.is_cached()


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_from_uri(uri_np: str, path_np: Path) -> None:
    shard = NumpySafetensorsShard.from_uri(uri_np)
    assert shard.equal(NumpySafetensorsShard(uri=uri_np, path=path_np))
    assert objects_are_equal(shard.get_data(), {"key1": np.ones((2, 3)), "key2": np.arange(5)})


def test_numpy_safetensors_shard_generate_uri_config(path_np: Path) -> None:
    assert NumpySafetensorsShard.generate_uri_config(path_np) == {
        KWARGS: {"path": path_np.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.NumpySafetensorsShardLoader"},
    }


def test_numpy_safetensors_shard_no_safetensors(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(RuntimeError, match="'safetensors' package is required but not installed."),
    ):
        NumpySafetensorsShard(uri="", path=tmp_path)


def test_numpy_safetensors_shard_no_numpy(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match="'numpy' package is required but not installed."),
    ):
        NumpySafetensorsShard(uri="", path=tmp_path)


###########################################
#     Tests for TorchSafetensorsShard     #
###########################################


@safetensors_available
@torch_available
def test_torch_safetensors_shard_str(uri: str, path: Path) -> None:
    assert str(TorchSafetensorsShard(uri=uri, path=path)).startswith("TorchSafetensorsShard(")


@safetensors_available
@torch_available
def test_torch_safetensors_shard_path(uri: str, path: Path) -> None:
    assert TorchSafetensorsShard(uri=uri, path=path).path == path


@safetensors_available
@torch_available
def test_torch_safetensors_shard_clear_not_initialized(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard(uri=uri, path=path)
    shard.clear()
    assert not shard.is_cached()


@safetensors_available
@torch_available
def test_torch_safetensors_shard_clear_is_cached(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard(uri=uri, path=path)
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": torch.ones(2, 3), "key2": torch.arange(5)}
    )
    assert shard.is_cached()
    shard.clear()
    assert not shard.is_cached()


@safetensors_available
@torch_available
def test_torch_safetensors_shard_equal_true(uri: str, path: Path) -> None:
    assert TorchSafetensorsShard(uri=uri, path=path).equal(
        TorchSafetensorsShard(uri=uri, path=path)
    )


@safetensors_available
@torch_available
def test_torch_safetensors_shard_equal_false_different_uri(uri: str, path: Path) -> None:
    assert not TorchSafetensorsShard(uri=uri, path=path).equal(
        TorchSafetensorsShard(uri="", path=path)
    )


@safetensors_available
@torch_available
def test_torch_safetensors_shard_equal_false_different_path(
    uri: str, path: Path, tmp_path: Path
) -> None:
    assert not TorchSafetensorsShard(uri=uri, path=path).equal(
        TorchSafetensorsShard(uri=uri, path=tmp_path)
    )


@safetensors_available
@torch_available
def test_torch_safetensors_shard_equal_false_different_type(uri: str, path: Path) -> None:
    assert not TorchSafetensorsShard(uri=uri, path=path).equal(42)


@safetensors_available
@torch_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_torch_safetensors_shard_equal_nan(tmp_path: Path, equal_nan: bool) -> None:
    shard = create_torch_safetensors_shard(
        data={"key1": torch.ones(2, 3), "key2": torch.tensor([1, 2, float("nan")])},
        uri=tmp_path.joinpath("uri").as_uri(),
    )
    assert shard.equal(TorchSafetensorsShard.from_uri(uri=shard.get_uri()), equal_nan=equal_nan)


@safetensors_available
@torch_available
def test_torch_safetensors_shard_get_data(uri: str, path: Path) -> None:
    assert objects_are_equal(
        TorchSafetensorsShard(uri=uri, path=path).get_data(),
        {
            "key1": torch.ones(2, 3),
            "key2": torch.arange(5),
        },
    )


@safetensors_available
@torch_available
def test_torch_safetensors_shard_get_data_cache_false_not_cached(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard(uri=uri, path=path)
    assert not shard.is_cached()
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})
    assert not shard.is_cached()


@safetensors_available
@torch_available
def test_torch_safetensors_shard_get_data_cache_false_cached(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})
    assert shard.is_cached()


@safetensors_available
@torch_available
def test_torch_safetensors_shard_get_data_cache_true_not_cached(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard(uri=uri, path=path)
    assert not shard.is_cached()
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": torch.ones(2, 3), "key2": torch.arange(5)}
    )
    assert shard.is_cached()


@safetensors_available
@torch_available
def test_torch_safetensors_shard_get_data_cache_true_cached(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": torch.ones(2, 3), "key2": torch.arange(5)}
    )
    assert shard.is_cached()


@safetensors_available
@torch_available
def test_torch_safetensors_shard_get_data_multiple_calls(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard(uri=uri, path=path)
    assert objects_are_equal(
        shard.get_data(cache=True), {"key1": torch.ones(2, 3), "key2": torch.arange(5)}
    )
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})
    shard.get_data()["key1"].add_(1.0)
    assert objects_are_equal(
        shard.get_data(), {"key1": torch.full((2, 3), 2.0), "key2": torch.arange(5)}
    )


@safetensors_available
@torch_available
def test_torch_safetensors_shard_get_uri(uri: str, path: Path) -> None:
    assert TorchSafetensorsShard(uri=uri, path=path).get_uri() == uri


@safetensors_available
@torch_available
def test_torch_safetensors_shard_is_cached_false(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard(uri=uri, path=path)
    assert not shard.is_cached()


@safetensors_available
@torch_available
def test_torch_safetensors_shard_is_cached_true(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard(uri=uri, path=path)
    shard.get_data(cache=True)
    assert shard.is_cached()


@safetensors_available
@torch_available
def test_torch_safetensors_shard_from_uri(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard.from_uri(uri)
    assert shard.equal(TorchSafetensorsShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


def test_torch_safetensors_shard_generate_uri_config(path: Path) -> None:
    assert TorchSafetensorsShard.generate_uri_config(path) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.TorchSafetensorsShardLoader"},
    }


def test_torch_safetensors_shard_no_safetensors(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(RuntimeError, match="'safetensors' package is required but not installed."),
    ):
        TorchSafetensorsShard(uri="", path=tmp_path)


def test_torch_safetensors_shard_no_torch(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match="'torch' package is required but not installed."),
    ):
        TorchSafetensorsShard(uri="", path=tmp_path)


####################################################
#     Tests for create_numpy_safetensors_shard     #
####################################################


@safetensors_available
@numpy_available
def test_create_numpy_safetensors_shard(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("my_uri.safetensors")
    shard = create_numpy_safetensors_shard(
        data={"key1": np.ones((2, 3)), "key2": np.arange(5)}, uri=uri
    )

    assert uri_file.is_file()
    assert load_json(uri_file) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.NumpySafetensorsShardLoader"},
    }
    assert shard.equal(NumpySafetensorsShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": np.ones((2, 3)), "key2": np.arange(5)})


@safetensors_available
@numpy_available
def test_create_numpy_safetensors_shard_with_data(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("data.safetensors")
    shard = create_numpy_safetensors_shard(
        data={"key1": np.ones((2, 3)), "key2": np.arange(5)}, uri=uri, path=path
    )

    assert load_json(uri_file) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.NumpySafetensorsShardLoader"},
    }
    assert shard.equal(NumpySafetensorsShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": np.ones((2, 3)), "key2": np.arange(5)})


####################################################
#     Tests for create_torch_safetensors_shard     #
####################################################


@safetensors_available
@torch_available
def test_create_torch_safetensors_shard(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("my_uri.safetensors")
    shard = create_torch_safetensors_shard(
        data={"key1": torch.ones(2, 3), "key2": torch.arange(5)}, uri=uri
    )

    assert uri_file.is_file()
    assert load_json(uri_file) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.TorchSafetensorsShardLoader"},
    }
    assert shard.equal(TorchSafetensorsShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


@safetensors_available
@torch_available
def test_create_torch_safetensors_shard_with_data(tmp_path: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    path = tmp_path.joinpath("data.safetensors")
    shard = create_torch_safetensors_shard(
        data={"key1": torch.ones(2, 3), "key2": torch.arange(5)}, uri=uri, path=path
    )

    assert load_json(uri_file) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.TorchSafetensorsShardLoader"},
    }
    assert shard.equal(TorchSafetensorsShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})
