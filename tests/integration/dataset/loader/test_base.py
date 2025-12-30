from __future__ import annotations

import pytest
from coola import objects_are_equal

from iden.dataset.loader import VanillaDatasetLoader
from iden.shard.loader import PickleShardLoader

#######################################
#     Tests for objects_are_equal     #
#######################################


def test_objects_are_equal_true() -> None:
    assert objects_are_equal(VanillaDatasetLoader(), VanillaDatasetLoader())


def test_objects_are_equal_false() -> None:
    assert not objects_are_equal(VanillaDatasetLoader(), PickleShardLoader())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_objects_are_equal_true_equal_nan(equal_nan: bool) -> None:
    assert objects_are_equal(VanillaDatasetLoader(), VanillaDatasetLoader(), equal_nan=equal_nan)
