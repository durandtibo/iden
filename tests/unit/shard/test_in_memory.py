from __future__ import annotations

from iden.shard import InMemoryShard

###################################
#     Tests for InMemoryShard     #
###################################


def test_in_memory_shard_repr() -> None:
    assert repr(InMemoryShard([1, 2, 3])) == "InMemoryShard()"


def test_in_memory_shard_str() -> None:
    assert str(InMemoryShard([1, 2, 3])) == "InMemoryShard()"


def test_in_memory_shard_clear() -> None:
    shard = InMemoryShard([1, 2, 3])
    assert shard.is_cached()
    shard.clear()
    assert shard.is_cached()


def test_in_memory_shard_equal_true() -> None:
    assert InMemoryShard([1, 2, 3]).equal(InMemoryShard([1, 2, 3]))


def test_in_memory_shard_equal_false_different_data() -> None:
    assert not InMemoryShard([1, 2, 3]).equal(InMemoryShard([1, 2, 4]))


def test_in_memory_shard_equal_equal_nan_true() -> None:
    assert InMemoryShard([1, 2, float("nan")]).equal(
        InMemoryShard([1, 2, float("nan")]), equal_nan=True
    )


def test_in_memory_shard_equal_equal_nan_false() -> None:
    assert not InMemoryShard([1, 2, float("nan")]).equal(
        InMemoryShard([1, 2, float("nan")]), equal_nan=False
    )


def test_in_memory_shard_equal_false_different_type() -> None:
    assert not InMemoryShard([1, 2, 3]).equal([1, 2, 4])


def test_in_memory_shard_equal_false_different_type_child() -> None:
    class Child(InMemoryShard): ...

    assert not InMemoryShard([1, 2, 3]).equal(Child([1, 2, 3]))


def test_in_memory_shard_get_data() -> None:
    assert InMemoryShard([1, 2, 3]).get_data() == [1, 2, 3]


def test_in_memory_shard_get_uri() -> None:
    assert InMemoryShard([1, 2, 3]).get_uri() is None


def test_in_memory_shard_is_cached() -> None:
    assert InMemoryShard([1, 2, 3]).is_cached()
