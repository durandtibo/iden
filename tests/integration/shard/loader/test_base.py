from __future__ import annotations

import pytest
from coola.equality import objects_are_equal

from iden.shard.loader import JsonShardLoader, PickleShardLoader

#######################################
#     Tests for objects_are_equal     #
#######################################


def test_objects_are_equal_true() -> None:
    assert objects_are_equal(JsonShardLoader(), JsonShardLoader())


def test_objects_are_equal_false() -> None:
    assert not objects_are_equal(JsonShardLoader(), PickleShardLoader())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_objects_are_equal_true_equal_nan(equal_nan: bool) -> None:
    assert objects_are_equal(JsonShardLoader(), JsonShardLoader(), equal_nan=equal_nan)
