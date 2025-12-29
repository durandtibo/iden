from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from coola.testing import torch_available

from iden.io import (
    LoaderRegistry,
    TextLoader,
    get_default_loader_registry,
    load,
    register_loaders,
    save_json,
    save_text,
)
from iden.testing import joblib_available, yaml_available

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_loader_registry, "_registry"):
        del get_default_loader_registry._registry
    yield
    if hasattr(get_default_loader_registry, "_registry"):
        del get_default_loader_registry._registry


##########################
#     Tests for load     #
##########################


def test_load_json(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json([1, 2, 3], path)
    assert load(path) == [1, 2, 3]


def test_load_txt(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    save_text("hello", path)
    assert load(path) == "hello"


def test_load_registry(tmp_path: Path) -> None:
    registry = LoaderRegistry({"json": TextLoader()})
    path = tmp_path.joinpath("data.json")
    save_json([1, 2, 3], path)
    assert load(path, registry) == "[1, 2, 3]"


######################################
#     Tests for register_loaders     #
######################################


def test_register_loaders_calls_registry() -> None:
    register_loaders({"longtext": TextLoader()})
    assert get_default_loader_registry().has_loader("longtext")


def test_register_loaders_with_exist_ok_true() -> None:
    register_loaders({"json": TextLoader()}, exist_ok=True)


def test_register_loaders_with_exist_ok_false() -> None:
    with pytest.raises(RuntimeError, match="already registered"):
        register_loaders({"json": TextLoader()}, exist_ok=False)


#################################################
#     Tests for get_default_loader_registry     #
#################################################


def test_get_default_loader_registry_returns_registry() -> None:
    """Test that get_default_loader_registry returns a LoaderRegistry
    instance."""
    registry = get_default_loader_registry()
    assert isinstance(registry, LoaderRegistry)


def test_get_default_loader_registry_returns_singleton() -> None:
    """Test that get_default_loader_registry returns the same instance
    on multiple calls."""
    registry1 = get_default_loader_registry()
    registry2 = get_default_loader_registry()
    assert registry1 is registry2


def test_get_default_loader_registry() -> None:
    """Test that scalar types are registered with DefaultLoader."""
    registry = get_default_loader_registry()
    assert registry.has_loader("json")
    assert registry.has_loader("pkl")
    assert registry.has_loader("pickle")
    assert registry.has_loader("txt")


@joblib_available
def test_get_default_loader_registry_joblib() -> None:
    registry = get_default_loader_registry()
    assert registry.has_loader("joblib")


def test_get_default_loader_registry_no_joblib() -> None:
    with patch("iden.io.loading.is_joblib_available", lambda: False):
        registry = get_default_loader_registry()
        assert not registry.has_loader("joblib")


@torch_available
def test_get_default_loader_registry_torch() -> None:
    registry = get_default_loader_registry()
    assert registry.has_loader("pt")


def test_get_default_loader_registry_no_torch() -> None:
    with patch("iden.io.loading.is_torch_available", lambda: False):
        registry = get_default_loader_registry()
        assert not registry.has_loader("pt")


@yaml_available
def test_get_default_loader_registry_yaml() -> None:
    registry = get_default_loader_registry()
    assert registry.has_loader("yaml")
    assert registry.has_loader("yml")


def test_get_default_loader_registry_no_yaml() -> None:
    with patch("iden.io.loading.is_yaml_available", lambda: False):
        registry = get_default_loader_registry()
        assert not registry.has_loader("yaml")
        assert not registry.has_loader("yml")


def test_get_default_loader_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the registry persist across calls."""
    registry1 = get_default_loader_registry()
    assert not registry1.has_loader("longtext")
    registry1.register("longtext", TextLoader())
    assert registry1.has_loader("longtext")

    # Get registry again
    registry2 = get_default_loader_registry()
    assert registry1 is registry2
    assert registry2.has_loader("longtext")
