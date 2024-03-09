from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal

from iden.shard import JsonShard, create_json_shard
from iden.shard.collection import ShardMapping
from iden.shard.exceptions import ShardExistsError, ShardNotFoundError

if TYPE_CHECKING:
    from pathlib import Path

    from iden.shard import BaseShard


@pytest.fixture(scope="module")
def path_shard(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("shards")


@pytest.fixture(scope="module")
def shard(path_shard: Path) -> BaseShard:
    return create_json_shard([10, 11], uri=path_shard.joinpath("uri4").as_uri())


@pytest.fixture(scope="module")
def shards(path_shard: Path) -> dict[str, BaseShard]:
    return {
        "001": create_json_shard([1, 2, 3], uri=path_shard.joinpath("uri1").as_uri()),
        "002": create_json_shard([4, 5, 6, 7], uri=path_shard.joinpath("uri2").as_uri()),
        "003": create_json_shard([8], uri=path_shard.joinpath("uri3").as_uri()),
    }


##################################
#     Tests for ShardMapping     #
##################################


def test_shard_mapping_len(shards: dict[str, BaseShard]) -> None:
    assert len(ShardMapping(shards)) == 3


def test_shard_mapping_repr(shards: dict[str, BaseShard]) -> None:
    assert repr(ShardMapping(shards)).startswith("ShardMapping(")


def test_shard_mapping_str(shards: dict[str, BaseShard]) -> None:
    assert str(ShardMapping(shards)).startswith("ShardMapping(")


def test_shard_mapping_add_shard(
    shards: dict[str, BaseShard], shard: BaseShard, path_shard: Path
) -> None:
    mapping = ShardMapping(shards)
    mapping.add_shard("004", shard)
    assert mapping.equal(
        ShardMapping(
            {
                "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
                "002": JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
                "003": JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
                "004": JsonShard.from_uri(uri=path_shard.joinpath("uri4").as_uri()),
            }
        )
    )


def test_shard_mapping_add_shard_replace_ok_false(
    shards: dict[str, BaseShard], shard: BaseShard
) -> None:
    mapping = ShardMapping(shards)
    with pytest.raises(ShardExistsError, match="`003` is already used to register a shard"):
        mapping.add_shard("003", shard)


def test_shard_mapping_add_shard_replace_ok_true(
    shards: dict[str, BaseShard], shard: BaseShard, path_shard: Path
) -> None:
    mapping = ShardMapping(shards)
    mapping.add_shard("003", shard, replace_ok=True)
    assert mapping.equal(
        ShardMapping(
            {
                "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
                "002": JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
                "003": JsonShard.from_uri(uri=path_shard.joinpath("uri4").as_uri()),
            }
        )
    )


def test_shard_mapping_equal_true(shards: dict[str, BaseShard]) -> None:
    assert ShardMapping(shards).equal(ShardMapping(shards))


def test_shard_mapping_equal_false_different_shards(shards: dict[str, BaseShard]) -> None:
    assert not ShardMapping(shards).equal(ShardMapping({}))


def test_shard_mapping_equal_false_different_type(shards: dict[str, BaseShard]) -> None:
    assert not ShardMapping(shards).equal({})


def test_shard_mapping_get_shard(shards: dict[str, BaseShard], path_shard: Path) -> None:
    mapping = ShardMapping(shards)
    assert mapping.get_shard("001").equal(
        JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri())
    )
    assert mapping.get_shard("002").equal(
        JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri())
    )
    assert mapping.get_shard("003").equal(
        JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri())
    )


def test_shard_mapping_get_shard_missing() -> None:
    mapping = ShardMapping()
    with pytest.raises(ShardNotFoundError, match="shard `missing` does not exist"):
        mapping.get_shard("missing")


def test_shard_mapping_get_shards(shards: dict[str, BaseShard], path_shard: Path) -> None:
    assert objects_are_equal(
        ShardMapping(shards).get_shards(),
        {
            "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            "002": JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            "003": JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        },
    )


def test_shard_mapping_get_shards_empty() -> None:
    assert objects_are_equal(ShardMapping().get_shards(), {})


def test_shard_mapping_has_shard_true(shards: dict[str, BaseShard]) -> None:
    mapping = ShardMapping(shards)
    assert mapping.has_shard("001")
    assert mapping.has_shard("002")
    assert mapping.has_shard("003")


def test_shard_mapping_has_shard_false() -> None:
    assert not ShardMapping().has_shard("missing")


def test_shard_mapping_remove_shard(shards: dict[str, BaseShard], path_shard: Path) -> None:
    mapping = ShardMapping(shards)
    mapping.remove_shard("002")
    assert mapping.equal(
        ShardMapping(
            {
                "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
                "003": JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
            }
        )
    )


def test_shard_mapping_remove_shard_missing() -> None:
    mapping = ShardMapping()
    with pytest.raises(ShardNotFoundError, match="shard `missing` does not exist"):
        mapping.remove_shard("missing")
