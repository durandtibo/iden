from __future__ import annotations

from typing import TYPE_CHECKING

from coola import objects_are_equal

from tests.unit.dataset.test_vanilla import create_dataset

if TYPE_CHECKING:
    from pathlib import Path


#######################################
#     Tests for objects_are_equal     #
#######################################


def test_objects_are_equal_true(tmp_path: Path) -> None:
    assert objects_are_equal(
        create_dataset(path=tmp_path, data=[1, 2, 3]), create_dataset(path=tmp_path, data=[1, 2, 3])
    )


def test_objects_are_equal_false(tmp_path: Path) -> None:
    assert not objects_are_equal(
        create_dataset(path=tmp_path, data=[1, 2, 3]), create_dataset(path=tmp_path, data=[])
    )


def test_objects_are_equal_true_equal_nan(tmp_path: Path) -> None:
    assert objects_are_equal(
        create_dataset(path=tmp_path, data=[1, 2, 3, float("nan")]),
        create_dataset(path=tmp_path, data=[1, 2, 3, float("nan")]),
        equal_nan=True,
    )


def test_objects_are_equal_false_equal_nan(tmp_path: Path) -> None:
    assert not objects_are_equal(
        create_dataset(path=tmp_path, data=[1, 2, 3, float("nan")]),
        create_dataset(path=tmp_path, data=[1, 2, 3, float("nan")]),
    )
