from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from coola.equality import objects_are_equal
from coola.testing.fixtures import torch_available

from iden.data.generator import DataGenerator
from iden.shard import TorchShard
from iden.shard.generator import TorchShardGenerator

if TYPE_CHECKING:
    from pathlib import Path

#########################################
#     Tests for TorchShardGenerator     #
#########################################


@torch_available
def test_torch_shard_generator_repr(tmp_path: Path) -> None:
    assert repr(
        TorchShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("TorchShardGenerator(")


@torch_available
def test_torch_shard_generator_str(tmp_path: Path) -> None:
    assert str(
        TorchShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("TorchShardGenerator(")


@torch_available
def test_torch_shard_generator_equal_true(tmp_path: Path) -> None:
    generator1 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert generator1.equal(generator2)


@torch_available
def test_torch_shard_generator_equal_false_different_data(tmp_path: Path) -> None:
    generator1 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = TorchShardGenerator(
        data=DataGenerator([]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert not generator1.equal(generator2)


@torch_available
def test_torch_shard_generator_equal_false_different_path_uri(tmp_path: Path) -> None:
    generator1 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("other/uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert not generator1.equal(generator2)


@torch_available
def test_torch_shard_generator_equal_false_different_path_shard(tmp_path: Path) -> None:
    generator1 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("other/shard"),
    )
    assert not generator1.equal(generator2)


@torch_available
def test_torch_shard_generator_equal_false_different_type(tmp_path: Path) -> None:
    generator = TorchShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert not generator.equal(42)


@torch_available
def test_torch_shard_generator_equal_false_different_type_child(tmp_path: Path) -> None:
    class Child(TorchShardGenerator): ...

    generator1 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = Child(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert not generator1.equal(generator2)


@torch_available
def test_torch_shard_generator_equal_true_equal_nan(tmp_path: Path) -> None:
    generator1 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3, float("nan")]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3, float("nan")]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert generator1.equal(generator2, equal_nan=True)


@torch_available
def test_torch_shard_generator_equal_false_equal_nan(tmp_path: Path) -> None:
    generator1 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3, float("nan")]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = TorchShardGenerator(
        data=DataGenerator([1, 2, 3, float("nan")]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert not generator1.equal(generator2)


@torch_available
def test_torch_shard_generator_generate(tmp_path: Path) -> None:
    generator = TorchShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    shard = generator.generate("000001")
    assert shard.equal(
        TorchShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.pt"),
        )
    )
    assert objects_are_equal(shard.get_data(), [1, 2, 3])


def test_torch_shard_generator_no_torch(tmp_path: Path) -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."),
    ):
        TorchShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
