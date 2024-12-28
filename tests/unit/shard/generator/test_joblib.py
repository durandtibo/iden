from __future__ import annotations

from typing import TYPE_CHECKING

from coola import objects_are_equal

from iden.data.generator import DataGenerator
from iden.shard import JoblibShard
from iden.shard.generator import JoblibShardGenerator
from iden.testing import joblib_available

if TYPE_CHECKING:
    from pathlib import Path

##########################################
#     Tests for JoblibShardGenerator     #
##########################################


@joblib_available
def test_joblib_shard_generator_repr(tmp_path: Path) -> None:
    assert repr(
        JoblibShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("JoblibShardGenerator(")


@joblib_available
def test_joblib_shard_generator_str(tmp_path: Path) -> None:
    assert str(
        JoblibShardGenerator(
            data=DataGenerator([1, 2, 3]),
            path_uri=tmp_path.joinpath("uri"),
            path_shard=tmp_path.joinpath("shard"),
        )
    ).startswith("JoblibShardGenerator(")


@joblib_available
def test_joblib_shard_generator_generate(tmp_path: Path) -> None:
    generator = JoblibShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    shard = generator.generate("000001")
    assert shard.equal(
        JoblibShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.joblib"),
        )
    )
    assert objects_are_equal(shard.get_data(), [1, 2, 3])
