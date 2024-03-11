from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from coola.testing import torch_available

from iden.io import (
    AutoFileLoader,
    JsonLoader,
    PickleLoader,
    TextLoader,
    TorchLoader,
    YamlLoader,
    save_json,
    save_text,
)
from iden.testing import yaml_available

if TYPE_CHECKING:
    from pathlib import Path

####################################
#     Tests for AutoFileLoader     #
####################################


def test_auto_file_loader_str() -> None:
    assert str(AutoFileLoader()).startswith("AutoFileLoader(")


@patch.dict(AutoFileLoader.registry, {}, clear=True)
def test_auto_file_loader_add_loader() -> None:
    loader = AutoFileLoader()
    text_loader = TextLoader()
    loader.add_loader("text", text_loader)
    assert isinstance(loader.registry["text"], TextLoader)
    assert loader.registry["text"] is text_loader


@patch.dict(AutoFileLoader.registry, {}, clear=True)
def test_auto_file_loader_add_loader_duplicate_exist_ok_true() -> None:
    loader = AutoFileLoader()
    text_loader = TextLoader()
    loader.add_loader("text", JsonLoader())
    loader.add_loader("text", text_loader, exist_ok=True)
    assert isinstance(loader.registry["text"], TextLoader)
    assert loader.registry["text"] is text_loader


@patch.dict(AutoFileLoader.registry, {}, clear=True)
def test_auto_file_loader_add_loader_duplicate_exist_ok_false() -> None:
    loader = AutoFileLoader()
    text_loader = TextLoader()
    loader.add_loader("text", JsonLoader())
    with pytest.raises(
        RuntimeError, match="A loader .* is already registered for the file extension"
    ):
        loader.add_loader("text", text_loader)


def test_auto_file_loader_has_loader_true() -> None:
    assert AutoFileLoader().has_loader("txt")


def test_auto_file_loader_has_loader_false() -> None:
    assert not AutoFileLoader().has_loader("newtxt")


def test_auto_file_loader_find_loader_txt() -> None:
    assert isinstance(AutoFileLoader().find_loader("txt"), TextLoader)


def test_auto_file_loader_find_loader_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Incorrect extension:"):
        AutoFileLoader().find_loader("newtxt")


def test_auto_file_loader_load_json(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json([1, 2, 3], path)
    assert AutoFileLoader().load(path) == [1, 2, 3]


def test_auto_file_loader_load_txt(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    save_text("hello", path)
    assert AutoFileLoader().load(path) == "hello"


def test_auto_file_loader_registry_default() -> None:
    assert len(AutoFileLoader.registry) >= 4
    assert isinstance(AutoFileLoader.registry["json"], JsonLoader)
    assert isinstance(AutoFileLoader.registry["pickle"], PickleLoader)
    assert isinstance(AutoFileLoader.registry["pkl"], PickleLoader)
    assert isinstance(AutoFileLoader.registry["txt"], TextLoader)


@torch_available
def test_auto_file_loader_registry_torch() -> None:
    assert isinstance(AutoFileLoader.registry["pt"], TorchLoader)


@yaml_available
def test_auto_file_loader_registry_yaml() -> None:
    assert isinstance(AutoFileLoader.registry["yaml"], YamlLoader)
    assert isinstance(AutoFileLoader.registry["yml"], YamlLoader)
