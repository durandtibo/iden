from __future__ import annotations

from typing import TYPE_CHECKING

from coola import objects_are_equal

from iden.data.generator import DataGenerator
from iden.shard.generator import JsonShardGenerator

if TYPE_CHECKING:
    from pathlib import Path

#######################################
#     Tests for objects_are_equal     #
#######################################


def test_objects_are_equal_true(tmp_path: Path) -> None:
    generator1 = JsonShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = JsonShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert objects_are_equal(generator1, generator2)


def test_objects_are_equal_false(tmp_path: Path) -> None:
    generator1 = JsonShardGenerator(
        data=DataGenerator([1, 2, 3]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    generator2 = JsonShardGenerator(
        data=DataGenerator([]),
        path_uri=tmp_path.joinpath("uri"),
        path_shard=tmp_path.joinpath("shard"),
    )
    assert not objects_are_equal(generator1, generator2)
