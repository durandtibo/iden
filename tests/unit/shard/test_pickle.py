from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iden.shard import PickleShard
from iden.utils.io import save_pickle

if TYPE_CHECKING:
    from pathlib import Path

#################################
#     Tests for PickleShard     #
#################################


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("tmp").joinpath("data.pkl")
    save_pickle([1, 2, 3], p)
    return p


def test_pickle_shard_str(path: Path) -> None:
    assert str(PickleShard(path)).startswith("PickleShard(")


def test_pickle_shard_get_data(path: Path) -> None:
    assert PickleShard(path).get_data() == [1, 2, 3]


def test_pickle_shard_get_uri(path: Path) -> None:
    assert PickleShard(path).get_uri() == path.as_uri()
