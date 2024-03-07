from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from coola import objects_are_equal
from coola.testing import torch_available
from coola.utils import is_torch_available
from objectory import OBJECT_TARGET

from iden.constants import KWARGS, LOADER
from iden.io import load_json
from iden.shard import TorchSafetensorsShard, create_torch_safetensors_shard
from iden.testing import safetensors_available

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
def test_torch_safetensors_shard_get_data_multiple_calls(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard(uri=uri, path=path)
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})
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
def test_torch_safetensors_shard_from_uri(uri: str, path: Path) -> None:
    shard = TorchSafetensorsShard.from_uri(uri)
    assert shard.equal(TorchSafetensorsShard(uri=uri, path=path))
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


def test_torch_safetensors_shard_generate_uri_config(path: Path) -> None:
    assert TorchSafetensorsShard.generate_uri_config(path) == {
        KWARGS: {"path": path.as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.TorchSafetensorsShardLoader"},
    }


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
