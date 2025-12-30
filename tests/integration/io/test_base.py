from __future__ import annotations

import pytest
from coola import objects_are_equal

from iden.io import JsonLoader, JsonSaver, PickleLoader, PickleSaver

#######################################
#     Tests for objects_are_equal     #
#######################################


def test_objects_are_equal_true_loader() -> None:
    assert objects_are_equal(JsonLoader(), JsonLoader())


def test_objects_are_equal_false_loader() -> None:
    assert not objects_are_equal(JsonLoader(), PickleLoader())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_objects_are_equal_true_equal_nan_loader(equal_nan: bool) -> None:
    assert objects_are_equal(JsonLoader(), JsonLoader(), equal_nan=equal_nan)


def test_objects_are_equal_true_saver() -> None:
    assert objects_are_equal(JsonSaver(), JsonSaver())


def test_objects_are_equal_false_saver() -> None:
    assert not objects_are_equal(JsonSaver(), PickleSaver())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_objects_are_equal_true_equal_nan_saver(equal_nan: bool) -> None:
    assert objects_are_equal(JsonSaver(), JsonSaver(), equal_nan=equal_nan)
