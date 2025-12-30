from __future__ import annotations

from coola import objects_are_equal

from iden.data.generator import DataGenerator

#######################################
#     Tests for objects_are_equal     #
#######################################


def test_objects_are_equal_true() -> None:
    assert objects_are_equal(DataGenerator([1, 2, 3]), DataGenerator([1, 2, 3]))


def test_objects_are_equal_false() -> None:
    assert not objects_are_equal(DataGenerator([1, 2, 3]), DataGenerator([]))


def test_objects_are_equal_true_equal_nan() -> None:
    assert objects_are_equal(
        DataGenerator([1, 2, 3, float("nan")]),
        DataGenerator([1, 2, 3, float("nan")]),
        equal_nan=True,
    )


def test_objects_are_equal_false_equal_nan() -> None:
    assert not objects_are_equal(
        DataGenerator([1, 2, 3, float("nan")]), DataGenerator([1, 2, 3, float("nan")])
    )
