from __future__ import annotations

from iden.shard import InMemoryShard

###################################
#     Tests for InMemoryShard     #
###################################


def test_in_memory_shard_str() -> None:
    assert str(InMemoryShard([1, 2, 3])) == "InMemoryShard()"


def test_in_memory_shard_equal_true() -> None:
    assert InMemoryShard([1, 2, 3]).equal(InMemoryShard([1, 2, 3]))


def test_in_memory_shard_equal_false_different_data() -> None:
    assert not InMemoryShard([1, 2, 3]).equal(InMemoryShard([1, 2, 4]))


def test_in_memory_shard_equal_false_different_type() -> None:
    assert not InMemoryShard([1, 2, 3]).equal([1, 2, 4])


def test_in_memory_shard_get_data() -> None:
    assert InMemoryShard([1, 2, 3]).get_data() == [1, 2, 3]


def test_in_memory_shard_get_uri() -> None:
    assert InMemoryShard([1, 2, 3]).get_uri() is None
