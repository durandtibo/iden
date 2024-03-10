from __future__ import annotations

from typing import TYPE_CHECKING

from coola import objects_are_equal

from iden.shard import (
    JsonShard,
    create_json_shard,
    get_dict_uris,
    get_list_uris,
    sort_by_uri,
)

if TYPE_CHECKING:
    from pathlib import Path


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
