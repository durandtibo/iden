from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola import objects_are_equal
from objectory import OBJECT_TARGET

from iden.constants import LOADER, SHARDS
from iden.io import load_json
from iden.shard import JsonShard, ShardDict, create_json_shard, create_shard_dict
from iden.shard.exceptions import ShardNotFoundError

if TYPE_CHECKING:
    from pathlib import Path

    from iden.shard import BaseShard


@pytest.fixture(scope="module")
def path_shard(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("shards")


@pytest.fixture(scope="module")
def shards(path_shard: Path) -> dict[str, BaseShard]:
    return {
        "001": create_json_shard([1, 2, 3], uri=path_shard.joinpath("uri1").as_uri()),
        "002": create_json_shard([4, 5, 6, 7], uri=path_shard.joinpath("uri2").as_uri()),
        "003": create_json_shard([8], uri=path_shard.joinpath("uri3").as_uri()),
    }


@pytest.fixture(scope="module")
def uri(tmp_path_factory: pytest.TempPathFactory, shards: dict[str, BaseShard]) -> str:
    uri_ = tmp_path_factory.mktemp("tmp").joinpath("uri").as_uri()
    create_shard_dict(shards=shards, uri=uri_)
    return uri_


###############################
#     Tests for ShardDict     #
###############################


def test_shard_dict_len(uri: str, shards: dict[str, BaseShard]) -> None:
    assert len(ShardDict(uri=uri, shards=shards)) == 3


def test_shard_dict_contain_true(uri: str, shards: dict[str, BaseShard]) -> None:
    sd = ShardDict(uri=uri, shards=shards)
    assert "001" in sd
    assert "002" in sd
    assert "003" in sd


def test_shard_dict_contain_false(uri: str) -> None:
    sd = ShardDict(uri=uri, shards={})
    assert "missing" not in sd


def test_shard_dict_getitem(uri: str, shards: dict[str, BaseShard], path_shard: Path) -> None:
    sd = ShardDict(uri=uri, shards=shards)
    assert sd["001"].equal(JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()))
    assert sd["002"].equal(JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()))
    assert sd["003"].equal(JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()))


def test_shard_dict_getitem_missing(uri: str) -> None:
    sd = ShardDict(uri=uri, shards={})
    with pytest.raises(KeyError, match="'missing'"):
        _ = sd["missing"]


def test_shard_dict_repr(uri: str, shards: dict[str, BaseShard]) -> None:
    assert repr(ShardDict(uri=uri, shards=shards)).startswith("ShardDict(")


def test_shard_dict_str(uri: str, shards: dict[str, BaseShard]) -> None:
    assert str(ShardDict(uri=uri, shards=shards)).startswith("ShardDict(")


def test_shard_dict_equal_true(uri: str, shards: dict[str, BaseShard]) -> None:
    assert ShardDict(uri=uri, shards=shards).equal(ShardDict(uri=uri, shards=shards))


def test_shard_dict_equal_false_different_uris(uri: str, shards: dict[str, BaseShard]) -> None:
    assert not ShardDict(uri=uri, shards=shards).equal(ShardDict(uri=uri + "123", shards=shards))


def test_shard_dict_equal_false_different_shards(uri: str, shards: dict[str, BaseShard]) -> None:
    assert not ShardDict(uri=uri, shards=shards).equal(ShardDict(uri=uri, shards={}))


def test_shard_dict_equal_false_different_type(uri: str, shards: dict[str, BaseShard]) -> None:
    assert not ShardDict(uri=uri, shards=shards).equal({})


def test_shard_dict_get_data(uri: str, shards: dict[str, BaseShard], path_shard: Path) -> None:
    assert objects_are_equal(
        ShardDict(uri=uri, shards=shards).get_data(),
        {
            "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            "002": JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            "003": JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        },
    )


def test_shard_dict_get_data_empty(uri: str) -> None:
    assert objects_are_equal(ShardDict(uri=uri, shards={}).get_data(), {})


def test_shard_dict_get_shard(uri: str, shards: dict[str, BaseShard], path_shard: Path) -> None:
    sd = ShardDict(uri=uri, shards=shards)
    assert sd.get_shard("001").equal(JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()))
    assert sd.get_shard("002").equal(JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()))
    assert sd.get_shard("003").equal(JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()))


def test_shard_dict_get_shard_missing(uri: str) -> None:
    sd = ShardDict(uri=uri, shards={})
    with pytest.raises(ShardNotFoundError, match="shard `missing` does not exist"):
        sd.get_shard("missing")


def test_shard_dict_get_shard_ids(uri: str, shards: dict[str, BaseShard]) -> None:
    assert ShardDict(uri=uri, shards=shards).get_shard_ids() == {"001", "002", "003"}


def test_shard_dict_get_shard_ids_empty(uri: str) -> None:
    assert ShardDict(uri=uri, shards={}).get_shard_ids() == set()


def test_shard_dict_has_shard_true(uri: str, shards: dict[str, BaseShard]) -> None:
    sd = ShardDict(uri=uri, shards=shards)
    assert sd.has_shard("001")
    assert sd.has_shard("002")
    assert sd.has_shard("003")


def test_shard_dict_has_shard_false(uri: str) -> None:
    assert not ShardDict(uri=uri, shards={}).has_shard("missing")


def test_shard_dict_from_uri(uri: str, shards: dict[str, BaseShard], path_shard: Path) -> None:
    shard = ShardDict.from_uri(uri)
    assert shard.equal(ShardDict(uri=uri, shards=shards))
    assert objects_are_equal(
        shard.get_data(),
        {
            "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            "002": JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            "003": JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        },
    )


def test_shard_dict_generate_uri_config(shards: dict[str, BaseShard], path_shard: Path) -> None:
    assert ShardDict.generate_uri_config(shards) == {
        SHARDS: {
            "001": path_shard.joinpath("uri1").as_uri(),
            "002": path_shard.joinpath("uri2").as_uri(),
            "003": path_shard.joinpath("uri3").as_uri(),
        },
        LOADER: {OBJECT_TARGET: "iden.shard.loader.ShardDictLoader"},
    }


#######################################
#     Tests for create_shard_dict     #
#######################################


def test_create_shard_dict(tmp_path: Path, shards: dict[str, BaseShard], path_shard: Path) -> None:
    uri_file = tmp_path.joinpath("my_uri")
    uri = uri_file.as_uri()
    shard = create_shard_dict(shards=shards, uri=uri)

    assert uri_file.is_file()
    assert objects_are_equal(
        load_json(uri_file),
        {
            SHARDS: {
                "001": path_shard.joinpath("uri1").as_uri(),
                "002": path_shard.joinpath("uri2").as_uri(),
                "003": path_shard.joinpath("uri3").as_uri(),
            },
            LOADER: {OBJECT_TARGET: "iden.shard.loader.ShardDictLoader"},
        },
    )
    assert shard.equal(ShardDict(uri=uri, shards=shards))
    assert objects_are_equal(
        shard.get_data(),
        {
            "001": JsonShard.from_uri(uri=path_shard.joinpath("uri1").as_uri()),
            "002": JsonShard.from_uri(uri=path_shard.joinpath("uri2").as_uri()),
            "003": JsonShard.from_uri(uri=path_shard.joinpath("uri3").as_uri()),
        },
    )
