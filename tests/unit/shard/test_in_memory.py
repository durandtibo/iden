from __future__ import annotations

from iden.shard import InMemoryShard

###################################
#     Tests for InMemoryShard     #
###################################


def test_in_memory_shard_str() -> None:
    assert str(InMemoryShard(1)) == "InMemoryShard()"


def test_in_memory_shard_get_data() -> None:
    assert InMemoryShard(1).get_data() == 1


def test_in_memory_shard_get_uri() -> None:
    assert InMemoryShard(1).get_uri() is None
