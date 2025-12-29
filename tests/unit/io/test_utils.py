from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

from iden.io.utils import generate_unique_tmp_path

if TYPE_CHECKING:
    from pathlib import Path


##############################################
#     Tests for generate_unique_tmp_path     #
##############################################


def test_generate_unique_tmp_path_no_suffix(tmp_path: Path) -> None:
    with patch("iden.io.utils.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert generate_unique_tmp_path(tmp_path.joinpath("data")) == tmp_path.joinpath(
            "data-a1b2c3"
        )


def test_generate_unique_tmp_path_one_suffix(tmp_path: Path) -> None:
    with patch("iden.io.utils.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert generate_unique_tmp_path(tmp_path.joinpath("data.json")) == tmp_path.joinpath(
            "data-a1b2c3.json"
        )


def test_generate_unique_tmp_path_two_suffixes(tmp_path: Path) -> None:
    with patch("iden.io.utils.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert generate_unique_tmp_path(tmp_path.joinpath("data.tar.gz")) == tmp_path.joinpath(
            "data-a1b2c3.tar.gz"
        )


def test_generate_unique_tmp_path_dir(tmp_path: Path) -> None:
    with patch("iden.io.utils.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert generate_unique_tmp_path(tmp_path.joinpath("data/")) == tmp_path.joinpath(
            "data-a1b2c3"
        )
