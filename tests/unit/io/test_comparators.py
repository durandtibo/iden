from __future__ import annotations

import logging

import pytest
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester

from iden.io import JsonLoader, JsonSaver, PickleSaver
from iden.io.comparators import IOEqualityComparator
from tests.unit.helpers import ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


IO_EQUAL = [
    pytest.param(
        ExamplePair(actual=JsonLoader(), expected=JsonLoader()),
        id="json loader",
    ),
    pytest.param(
        ExamplePair(actual=JsonSaver(), expected=JsonSaver()),
        id="json saver",
    ),
]
IO_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=PickleSaver(protocol=5),
            expected=PickleSaver(),
            expected_message="objects are not equal:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=JsonLoader(),
            expected=JsonSaver(),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
IO_EQUAL_TOLERANCE = [
    pytest.param(
        ExamplePair(actual=JsonLoader(), expected=JsonLoader(), atol=1.0),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(actual=JsonLoader(), expected=JsonLoader(), rtol=1.0),
        id="rtol=1",
    ),
]


##########################################
#     Tests for IOEqualityComparator     #
##########################################


def test_io_equality_comparator_str() -> None:
    assert str(IOEqualityComparator()) == "IOEqualityComparator()"


def test_io_equality_comparator__eq__true() -> None:
    assert IOEqualityComparator() == IOEqualityComparator()


def test_io_equality_comparator__eq__false() -> None:
    assert IOEqualityComparator() != 123


def test_io_equality_comparator__eq__false_child() -> None:
    class ChildComparator(IOEqualityComparator): ...

    assert IOEqualityComparator() != ChildComparator()


def test_io_equality_comparator_clone() -> None:
    op = IOEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_io_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = JsonLoader()
    assert IOEqualityComparator().equal(obj, obj, config)


@pytest.mark.parametrize("example", IO_EQUAL)
def test_io_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = IOEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", IO_EQUAL)
def test_io_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = IOEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", IO_NOT_EQUAL)
def test_io_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = IOEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", IO_NOT_EQUAL)
def test_io_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = IOEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_io_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        IOEqualityComparator().equal(
            actual=PickleSaver(protocol=float("nan")),
            expected=PickleSaver(protocol=float("nan")),
            config=config,
        )
        == equal_nan
    )
