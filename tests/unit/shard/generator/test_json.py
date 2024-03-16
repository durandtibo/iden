from __future__ import annotations

from typing import TYPE_CHECKING

from coola import objects_are_equal

from iden.shard import JsonShard
from iden.shard.generator import JsonShardGenerator

if TYPE_CHECKING:
    from pathlib import Path

########################################
#     Tests for JsonShardGenerator     #
########################################


def test_json_shard_generator_repr(tmp_path: Path) -> None:
    assert repr(
        JsonShardGenerator(
            data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
        )
    ).startswith("JsonShardGenerator(")


def test_json_shard_generator_str(tmp_path: Path) -> None:
    assert str(
        JsonShardGenerator(
            data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
        )
    ).startswith("JsonShardGenerator(")


def test_json_shard_generator_generate(tmp_path: Path) -> None:
    generator = JsonShardGenerator(
        data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
    )
    shard = generator.generate("000001")
    assert shard.equal(
        JsonShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.json"),
        )
    )
    assert objects_are_equal(shard.get_data(), [1, 2, 3])
