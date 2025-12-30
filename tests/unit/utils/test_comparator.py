from __future__ import annotations

import logging

import pytest
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester

from iden.io import JsonLoader, JsonSaver, PickleSaver
from iden.shard import InMemoryShard
from iden.utils.comparator import ObjectEqualityComparator
from tests.unit.helpers import ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


OBJECT_EQUAL = [
    pytest.param(
        ExamplePair(actual=JsonLoader(), expected=JsonLoader()),
        id="json loader",
    ),
    pytest.param(
        ExamplePair(actual=JsonSaver(), expected=JsonSaver()),
        id="json saver",
    ),
    pytest.param(
        ExamplePair(actual=InMemoryShard([]), expected=InMemoryShard([])),
        id="shard list empty",
    ),
    pytest.param(
        ExamplePair(actual=InMemoryShard([1, 2, 3]), expected=InMemoryShard([1, 2, 3])),
        id="shard list int",
    ),
    pytest.param(
        ExamplePair(
            actual=InMemoryShard([1.0, 2.0, 3.0, 4.0]), expected=InMemoryShard([1.0, 2.0, 3.0, 4.0])
        ),
        id="shard list float",
    ),
    pytest.param(
        ExamplePair(actual=InMemoryShard(["a", "b", "c"]), expected=InMemoryShard(["a", "b", "c"])),
        id="shard list str",
    ),
]
OBJECT_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=PickleSaver(protocol=5),
            expected=PickleSaver(),
            expected_message="objects are not equal:",
        ),
        id="saver different values",
    ),
    pytest.param(
        ExamplePair(
            actual=JsonSaver(),
            expected=JsonLoader(),
            expected_message="objects have different types:",
        ),
        id="saver different types",
    ),
    pytest.param(
        ExamplePair(
            actual=InMemoryShard([1, 2, 3]),
            expected=InMemoryShard([1, 2, 4]),
            expected_message="objects are not equal:",
        ),
        id="shard different values",
    ),
    pytest.param(
        ExamplePair(
            actual=InMemoryShard([1, 2, 3]),
            expected=InMemoryShard([1, 2, 3, 4]),
            expected_message="objects are not equal:",
        ),
        id="shard different iden sizes",
    ),
    pytest.param(
        ExamplePair(
            actual=InMemoryShard([1, 2, 3]),
            expected=[1, 2, 3],
            expected_message="objects have different types:",
        ),
        id="shard different types",
    ),
]
OBJECT_EQUAL_TOLERANCE = [
    pytest.param(
        ExamplePair(actual=JsonLoader(), expected=JsonLoader(), atol=1.0),
        id="loader atol=1",
    ),
    pytest.param(
        ExamplePair(actual=JsonLoader(), expected=JsonLoader(), rtol=1.0),
        id="loader rtol=1",
    ),
    pytest.param(
        ExamplePair(actual=InMemoryShard([1, 2, 3]), expected=InMemoryShard([1, 2, 4]), atol=1.0),
        id="shard atol=1",
    ),
    pytest.param(
        ExamplePair(actual=InMemoryShard([1, 2, 3]), expected=InMemoryShard([1, 2, 4]), rtol=1.0),
        id="shard rtol=1",
    ),
]


##############################################
#     Tests for ObjectEqualityComparator     #
##############################################


def test_object_equality_comparator_repr() -> None:
    assert repr(ObjectEqualityComparator()) == "ObjectEqualityComparator()"


def test_object_equality_comparator_str() -> None:
    assert str(ObjectEqualityComparator()) == "ObjectEqualityComparator()"


def test_object_equality_comparator__eq__true() -> None:
    assert ObjectEqualityComparator() == ObjectEqualityComparator()


def test_object_equality_comparator__eq__false() -> None:
    assert ObjectEqualityComparator() != 123


def test_object_equality_comparator_clone() -> None:
    op = ObjectEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_object_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    iden = InMemoryShard([1, 2, 3])
    assert ObjectEqualityComparator().equal(iden, iden, config)


@pytest.mark.parametrize("example", OBJECT_EQUAL)
def test_object_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ObjectEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", OBJECT_EQUAL)
def test_object_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ObjectEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", OBJECT_NOT_EQUAL)
def test_object_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ObjectEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", OBJECT_NOT_EQUAL)
def test_object_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ObjectEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_object_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        ObjectEqualityComparator().equal(
            actual=InMemoryShard([1, 2, float("nan")]),
            expected=InMemoryShard([1, 2, float("nan")]),
            config=config,
        )
        == equal_nan
    )
