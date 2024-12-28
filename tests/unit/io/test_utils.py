from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

from coola.testing import torch_available

from iden.io import (
    JoblibLoader,
    JsonLoader,
    PickleLoader,
    TextLoader,
    TorchLoader,
    YamlLoader,
)
from iden.io.utils import generate_unique_tmp_path, get_loader_mapping
from iden.testing import joblib_available, yaml_available

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


########################################
#     Tests for get_loader_mapping     #
########################################


def test_get_loader_mapping() -> None:
    mapping = get_loader_mapping()
    assert len(mapping) >= 4
    assert isinstance(mapping["json"], JsonLoader)
    assert isinstance(mapping["pickle"], PickleLoader)
    assert isinstance(mapping["pkl"], PickleLoader)
    assert isinstance(mapping["txt"], TextLoader)


@joblib_available
def test_get_loader_mapping_joblib() -> None:
    mapping = get_loader_mapping()
    assert isinstance(mapping["joblib"], JoblibLoader)


@torch_available
def test_get_loader_mapping_torch() -> None:
    mapping = get_loader_mapping()
    assert isinstance(mapping["pt"], TorchLoader)


@yaml_available
def test_get_loader_mapping_yaml() -> None:
    mapping = get_loader_mapping()
    assert isinstance(mapping["yaml"], YamlLoader)
    assert isinstance(mapping["yml"], YamlLoader)
