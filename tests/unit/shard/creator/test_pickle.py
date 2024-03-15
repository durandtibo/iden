from __future__ import annotations

from typing import TYPE_CHECKING

from coola import objects_are_equal

from iden.shard import PickleShard
from iden.shard.creator import PickleShardCreator

if TYPE_CHECKING:
    from pathlib import Path

########################################
#     Tests for PickleShardCreator     #
########################################


def test_pickle_shard_creator_repr(tmp_path: Path) -> None:
    assert repr(
        PickleShardCreator(
            data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
        )
    ).startswith("PickleShardCreator(")


def test_pickle_shard_creator_str(tmp_path: Path) -> None:
    assert str(
        PickleShardCreator(
            data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
        )
    ).startswith("PickleShardCreator(")


def test_pickle_shard_creator_create(tmp_path: Path) -> None:
    creator = PickleShardCreator(
        data=[1, 2, 3], path_uri=tmp_path.joinpath("uri"), path_shard=tmp_path.joinpath("shard")
    )
    shard = creator.create("000001")
    assert shard.equal(
        PickleShard(
            uri=tmp_path.joinpath("uri/000001").as_uri(),
            path=tmp_path.joinpath("shard/000001.pkl"),
        )
    )
    assert objects_are_equal(shard.get_data(), [1, 2, 3])
