from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal

from iden.shard import JsonShard, create_json_shard
from iden.shard.collection import ShardDict
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


###############################
#     Tests for ShardDict     #
###############################


def test_shard_dict_len(shards: dict[str, BaseShard]) -> None:
    assert len(ShardDict(shards)) == 3


def test_shard_dict_repr(shards: dict[str, BaseShard]) -> None:
    assert repr(ShardDict(shards)).startswith("ShardDict(")


def test_shard_dict_str(shards: dict[str, BaseShard]) -> None:
    assert str(ShardDict(shards)).startswith("ShardDict(")


def test_shard_dict_add_shard(
    shards: dict[str, BaseShard], shard: BaseShard, path_shard: Path
) -> None:
    sd = ShardDict(shards)
    sd.add_shard("004", shard)
    assert sd.equal(
        ShardDict(
            {
                "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
                "002": JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
                "003": JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
                "004": JsonShard.from_uri(uri=path_shard.joinpath("uri4").as_uri()),
            }
        )
    )


def test_shard_dict_add_shard_replace_ok_false(
    shards: dict[str, BaseShard], shard: BaseShard
) -> None:
    sd = ShardDict(shards)
    with pytest.raises(ShardExistsError, match="`003` is already used to register a shard"):
        sd.add_shard("003", shard)


def test_shard_dict_add_shard_replace_ok_true(
    shards: dict[str, BaseShard], shard: BaseShard, path_shard: Path
) -> None:
    sd = ShardDict(shards)
    sd.add_shard("003", shard, replace_ok=True)
    assert sd.equal(
        ShardDict(
            {
                "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
                "002": JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
                "003": JsonShard.from_uri(uri=path_shard.joinpath("uri4").as_uri()),
            }
        )
    )


def test_shard_dict_equal_true(shards: dict[str, BaseShard]) -> None:
    assert ShardDict(shards).equal(ShardDict(shards))


def test_shard_dict_equal_false_different_shards(shards: dict[str, BaseShard]) -> None:
    assert not ShardDict(shards).equal(ShardDict({}))


def test_shard_dict_equal_false_different_type(shards: dict[str, BaseShard]) -> None:
    assert not ShardDict(shards).equal({})


def test_shard_dict_get_shard(shards: dict[str, BaseShard], path_shard: Path) -> None:
    sd = ShardDict(shards)
    assert sd.get_shard("001").equal(JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()))
    assert sd.get_shard("002").equal(JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()))
    assert sd.get_shard("003").equal(JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()))


def test_shard_dict_get_shard_missing() -> None:
    sd = ShardDict()
    with pytest.raises(ShardNotFoundError, match="shard `missing` does not exist"):
        sd.get_shard("missing")


def test_shard_dict_get_shards(shards: dict[str, BaseShard], path_shard: Path) -> None:
    assert objects_are_equal(
        ShardDict(shards).get_shards(),
        {
            "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            "002": JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            "003": JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        },
    )


def test_shard_dict_get_shards_empty() -> None:
    assert objects_are_equal(ShardDict().get_shards(), {})


def test_shard_dict_has_shard_true(shards: dict[str, BaseShard]) -> None:
    sd = ShardDict(shards)
    assert sd.has_shard("001")
    assert sd.has_shard("002")
    assert sd.has_shard("003")


def test_shard_dict_has_shard_false() -> None:
    assert not ShardDict().has_shard("missing")


def test_shard_dict_remove_shard(shards: dict[str, BaseShard], path_shard: Path) -> None:
    sd = ShardDict(shards)
    sd.remove_shard("002")
    assert sd.equal(
        ShardDict(
            {
                "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
                "003": JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
            }
        )
    )


def test_shard_dict_remove_shard_missing() -> None:
    sd = ShardDict()
    with pytest.raises(ShardNotFoundError, match="shard `missing` does not exist"):
        sd.remove_shard("missing")
