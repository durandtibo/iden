from __future__ import annotations

from typing import TYPE_CHECKING

from coola import objects_are_equal

from iden.data.generator import DataGenerator
from iden.shard import CloudpickleShard
from iden.shard.generator import CloudpickleShardGenerator
from iden.testing import cloudpickle_available

if TYPE_CHECKING:
    from pathlib import Path

###############################################
#     Tests for CloudpickleShardGenerator     #
###############################################


@cloudpickle_available
def test_cloudpickle_shard_generator_repr(tmp_path: Path) -> None:
    assert repr(
        CloudpickleShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("CloudpickleShardGenerator(")


@cloudpickle_available
def test_cloudpickle_shard_generator_str(tmp_path: Path) -> None:
    assert str(
        CloudpickleShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("CloudpickleShardGenerator(")


@cloudpickle_available
def test_cloudpickle_shard_generator_generate(tmp_path: Path) -> None:
    generator = CloudpickleShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    shard = generator.generate("000001")
    assert shard.equal(
        CloudpickleShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.pkl"),
        )
    )
    assert objects_are_equal(shard.get_data(), [1, 2, 3])
