from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from coola import objects_are_equal
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available

from iden.data.generator import DataGenerator
from iden.shard import NumpySafetensorsShard, TorchSafetensorsShard
from iden.shard.generator import (
    NumpySafetensorsShardGenerator,
    TorchSafetensorsShardGenerator,
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


####################################################
#     Tests for NumpySafetensorsShardGenerator     #
####################################################


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_generator_repr(tmp_path: Path) -> None:
    assert repr(
        NumpySafetensorsShardGenerator(
            data=DataGenerator({"key1": np.ones((2, 3)), "key2": np.arange(5)}),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("NumpySafetensorsShardGenerator(")


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_generator_str(tmp_path: Path) -> None:
    assert str(
        NumpySafetensorsShardGenerator(
            data=DataGenerator({"key1": np.ones((2, 3)), "key2": np.arange(5)}),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("NumpySafetensorsShardGenerator(")


@safetensors_available
@numpy_available
def test_numpy_safetensors_shard_generator_generate(tmp_path: Path) -> None:
    generator = NumpySafetensorsShardGenerator(
        data=DataGenerator({"key1": np.ones((2, 3)), "key2": np.arange(5)}),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    shard = generator.generate("000001")
    assert shard.equal(
        NumpySafetensorsShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.safetensors"),
        )
    )
    assert objects_are_equal(shard.get_data(), {"key1": np.ones((2, 3)), "key2": np.arange(5)})


def test_numpy_safetensors_shard_generator_no_safetensors(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(
            RuntimeError, match="`|'safetensors`|' package is required but not installed."
        ),
    ):
        NumpySafetensorsShardGenerator(
            data=DataGenerator({"key1": np.ones((2, 3)), "key2": np.arange(5)}),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )


def test_numpy_safetensors_shard_generator_no_numpy(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match="`|'numpy`|' package is required but not installed."),
    ):
        NumpySafetensorsShardGenerator(
            data=DataGenerator({"key1": np.ones((2, 3)), "key2": np.arange(5)}),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )


####################################################
#     Tests for TorchSafetensorsShardGenerator     #
####################################################


@safetensors_available
@torch_available
def test_torch_safetensors_shard_generator_repr(tmp_path: Path) -> None:
    assert repr(
        TorchSafetensorsShardGenerator(
            data=DataGenerator({"key1": torch.ones(2, 3), "key2": torch.arange(5)}),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("TorchSafetensorsShardGenerator(")


@safetensors_available
@torch_available
def test_torch_safetensors_shard_generator_str(tmp_path: Path) -> None:
    assert str(
        TorchSafetensorsShardGenerator(
            data=DataGenerator({"key1": torch.ones(2, 3), "key2": torch.arange(5)}),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("TorchSafetensorsShardGenerator(")


@safetensors_available
@torch_available
def test_torch_safetensors_shard_generator_generate(tmp_path: Path) -> None:
    generator = TorchSafetensorsShardGenerator(
        data=DataGenerator({"key1": torch.ones(2, 3), "key2": torch.arange(5)}),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    shard = generator.generate("000001")
    assert shard.equal(
        TorchSafetensorsShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.safetensors"),
        )
    )
    assert objects_are_equal(shard.get_data(), {"key1": torch.ones(2, 3), "key2": torch.arange(5)})


def test_torch_safetensors_shard_generator_no_safetensors(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(
            RuntimeError, match="`|'safetensors`|' package is required but not installed."
        ),
    ):
        TorchSafetensorsShardGenerator(
            data=DataGenerator({"key1": torch.ones(2, 3), "key2": torch.arange(5)}),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )


def test_torch_safetensors_shard_generator_no_torch(tmp_path: Path) -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: True),
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match="`|'torch`|' package is required but not installed."),
    ):
        TorchSafetensorsShardGenerator(
            data=DataGenerator({"key1": torch.ones(2, 3), "key2": torch.arange(5)}),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
