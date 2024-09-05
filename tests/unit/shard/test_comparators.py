from __future__ import annotations

import logging

import pytest
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester

from iden.shard import InMemoryShard
from iden.shard.comparators import ShardEqualityComparator
from tests.unit.helpers import ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


SHARD_EQUAL = [
    pytest.param(
        ExamplePair(actual=InMemoryShard([]), expected=InMemoryShard([])),
        id="list empty",
    ),
    pytest.param(
        ExamplePair(actual=InMemoryShard([1, 2, 3]), expected=InMemoryShard([1, 2, 3])),
        id="list int",
    ),
    pytest.param(
        ExamplePair(
            actual=InMemoryShard([1.0, 2.0, 3.0, 4.0]), expected=InMemoryShard([1.0, 2.0, 3.0, 4.0])
        ),
        id="list float",
    ),
    pytest.param(
        ExamplePair(actual=InMemoryShard(["a", "b", "c"]), expected=InMemoryShard(["a", "b", "c"])),
        id="list str",
    ),
]
SHARD_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=InMemoryShard([1, 2, 3]),
            expected=InMemoryShard([1, 2, 4]),
            expected_message="objects are not equal:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=InMemoryShard([1, 2, 3]),
            expected=InMemoryShard([1, 2, 3, 4]),
            expected_message="objects are not equal:",
        ),
        id="different shard sizes",
    ),
    pytest.param(
        ExamplePair(
            actual=InMemoryShard([1, 2, 3]),
            expected=[1, 2, 3],
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
SHARD_EQUAL_TOLERANCE = [
    pytest.param(
        ExamplePair(actual=InMemoryShard([1, 2, 3]), expected=InMemoryShard([1, 2, 4]), atol=1.0),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(actual=InMemoryShard([1, 2, 3]), expected=InMemoryShard([1, 2, 4]), atol=1.0),
        id="rtol=1",
    ),
]


#############################################
#     Tests for ShardEqualityComparator     #
#############################################


def test_shard_equality_comparator_str() -> None:
    assert str(ShardEqualityComparator()) == "ShardEqualityComparator()"


def test_shard_equality_comparator__eq__true() -> None:
    assert ShardEqualityComparator() == ShardEqualityComparator()


def test_shard_equality_comparator__eq__false() -> None:
    assert ShardEqualityComparator() != 123


def test_shard_equality_comparator_clone() -> None:
    op = ShardEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_shard_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    shard = InMemoryShard([1, 2, 3])
    assert ShardEqualityComparator().equal(shard, shard, config)


@pytest.mark.parametrize("example", SHARD_EQUAL)
def test_shard_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ShardEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", SHARD_EQUAL)
def test_shard_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ShardEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", SHARD_NOT_EQUAL)
def test_shard_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ShardEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", SHARD_NOT_EQUAL)
def test_shard_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ShardEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_shard_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        ShardEqualityComparator().equal(
            actual=InMemoryShard([1, 2, float("nan")]),
            expected=InMemoryShard([1, 2, float("nan")]),
            config=config,
        )
        == equal_nan
    )
