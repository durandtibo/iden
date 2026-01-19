from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola.equality import objects_are_equal

from iden.shard import (
    BaseShard,
    JsonShard,
    create_json_shard,
    get_dict_uris,
    get_list_uris,
    sort_by_uri,
)
from iden.shard.utils import ShardIterable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@pytest.fixture(scope="module")
def path_shard(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("shards")


@pytest.fixture(scope="module")
def shards(path_shard: Path) -> tuple[BaseShard, ...]:
    return (
        create_json_shard([1, 2, 3], uri=path_shard.joinpath("uri1").as_uri()),
        create_json_shard([4, 5, 6, 7], uri=path_shard.joinpath("uri2").as_uri()),
        create_json_shard([8], uri=path_shard.joinpath("uri3").as_uri()),
    )


###################################
#     Tests for ShardIterable     #
###################################


def test_shard_iterable_str() -> None:
    assert str(ShardIterable([])).startswith("ShardIterable(")


def test_shard_iterable_iter(shards: Iterable[BaseShard]) -> None:
    assert not any(shard.is_cached() for shard in shards)
    assert objects_are_equal(list(ShardIterable(shards)), [[1, 2, 3], [4, 5, 6, 7], [8]])
    assert not any(shard.is_cached() for shard in shards)


def test_shard_iterable_iter_empty() -> None:
    assert objects_are_equal(list(ShardIterable([])), [])


###################################
#     Tests for get_dict_uris     #
###################################


def test_get_dict_uris(tmp_path: Path) -> None:
    shards = {
        "001": create_json_shard([1, 2, 3], uri=tmp_path.joinpath("uri1").as_uri()),
        "002": create_json_shard([4, 5, 6, 7], uri=tmp_path.joinpath("uri2").as_uri()),
        "003": create_json_shard([8], uri=tmp_path.joinpath("uri3").as_uri()),
    }
    assert get_dict_uris(shards) == {
        "001": tmp_path.joinpath("uri1").as_uri(),
        "002": tmp_path.joinpath("uri2").as_uri(),
        "003": tmp_path.joinpath("uri3").as_uri(),
    }


def test_get_dict_uris_empty() -> None:
    assert get_dict_uris({}) == {}


###################################
#     Tests for get_list_uris     #
###################################


def test_get_list_uris(tmp_path: Path) -> None:
    shards = (
        create_json_shard([1, 2, 3], uri=tmp_path.joinpath("uri1").as_uri()),
        create_json_shard([4, 5, 6, 7], uri=tmp_path.joinpath("uri2").as_uri()),
        create_json_shard([8], uri=tmp_path.joinpath("uri3").as_uri()),
    )
    assert get_list_uris(shards) == [
        tmp_path.joinpath("uri1").as_uri(),
        tmp_path.joinpath("uri2").as_uri(),
        tmp_path.joinpath("uri3").as_uri(),
    ]


def test_get_list_uris_empty() -> None:
    assert get_list_uris([]) == []


#################################
#     Tests for sort_by_uri     #
#################################


def test_sort_by_uri(tmp_path: Path) -> None:
    shards = sort_by_uri(
        [
            create_json_shard([4, 5, 6, 7], uri=tmp_path.joinpath("uri2").as_uri()),
            create_json_shard([8], uri=tmp_path.joinpath("uri3").as_uri()),
            create_json_shard([1, 2, 3], uri=tmp_path.joinpath("uri1").as_uri()),
        ]
    )
    assert objects_are_equal(
        sort_by_uri(shards),
        [
            JsonShard.from_uri(tmp_path.joinpath("uri1").as_uri()),
            JsonShard.from_uri(tmp_path.joinpath("uri2").as_uri()),
            JsonShard.from_uri(tmp_path.joinpath("uri3").as_uri()),
        ],
    )


def test_sort_by_uri_reverse(tmp_path: Path) -> None:
    shards = sort_by_uri(
        [
            create_json_shard([4, 5, 6, 7], uri=tmp_path.joinpath("uri2").as_uri()),
            create_json_shard([8], uri=tmp_path.joinpath("uri3").as_uri()),
            create_json_shard([1, 2, 3], uri=tmp_path.joinpath("uri1").as_uri()),
        ]
    )
    assert objects_are_equal(
        sort_by_uri(shards, reverse=True),
        [
            JsonShard.from_uri(tmp_path.joinpath("uri3").as_uri()),
            JsonShard.from_uri(tmp_path.joinpath("uri2").as_uri()),
            JsonShard.from_uri(tmp_path.joinpath("uri1").as_uri()),
        ],
    )


def test_sort_by_uri_empty() -> None:
    assert sort_by_uri([]) == []
