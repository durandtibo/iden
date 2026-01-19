from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola.equality import objects_are_equal

from iden.io import (
    JsonLoader,
    LoaderRegistry,
    TextLoader,
    save_json,
    save_text,
)

if TYPE_CHECKING:
    from pathlib import Path

####################################
#     Tests for LoaderRegistry     #
####################################


def test_loader_registry_init_empty() -> None:
    assert LoaderRegistry()._registry == {}


def test_loader_registry_init_with_registry() -> None:
    assert objects_are_equal(
        LoaderRegistry({"text": TextLoader()})._registry, {"text": TextLoader()}
    )


def test_loader_registry_repr() -> None:
    assert repr(LoaderRegistry()).startswith("LoaderRegistry(")


def test_loader_registry_str() -> None:
    assert str(LoaderRegistry()).startswith("LoaderRegistry(")


def test_loader_registry_equal_true() -> None:
    assert LoaderRegistry().equal(LoaderRegistry())


def test_loader_registry_equal_false() -> None:
    assert not LoaderRegistry().equal(JsonLoader())


def test_loader_registry_equal_false_child() -> None:
    class Child(LoaderRegistry): ...

    assert not LoaderRegistry().equal(Child())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_loader_registry_equal_nan(equal_nan: bool) -> None:
    assert LoaderRegistry().equal(LoaderRegistry(), equal_nan=equal_nan)


def test_loader_registry_load_json(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json([1, 2, 3], path)
    assert LoaderRegistry({"json": JsonLoader(), "txt": TextLoader()}).load(path) == [1, 2, 3]


def test_loader_registry_load_txt(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    save_text("hello", path)
    assert LoaderRegistry({"json": JsonLoader(), "txt": TextLoader()}).load(path) == "hello"


def test_loader_registry_register() -> None:
    loader = LoaderRegistry()
    text_loader = TextLoader()
    loader.register("text", text_loader)
    assert objects_are_equal(loader._registry, {"text": TextLoader()})


def test_loader_registry_register_duplicate_exist_ok_true() -> None:
    loader = LoaderRegistry()
    text_loader = TextLoader()
    loader.register("text", JsonLoader())
    loader.register("text", text_loader, exist_ok=True)
    assert loader._registry == {"text": text_loader}


def test_loader_registry_register_duplicate_exist_ok_false() -> None:
    loader = LoaderRegistry()
    text_loader = TextLoader()
    loader.register("text", JsonLoader())
    with pytest.raises(RuntimeError, match=r"Loader .* already registered"):
        loader.register("text", text_loader)


def test_loader_registry_register_many() -> None:
    loader = LoaderRegistry()
    loader.register_many({"json": JsonLoader(), "txt": TextLoader()})
    assert objects_are_equal(loader._registry, {"json": JsonLoader(), "txt": TextLoader()})


def test_loader_registry_register_many_with_exist_ok() -> None:
    loader = LoaderRegistry()
    loader.register("json", JsonLoader())
    loader.register_many({"json": JsonLoader(), "txt": TextLoader()}, exist_ok=True)
    assert objects_are_equal(loader._registry, {"json": JsonLoader(), "txt": TextLoader()})


def test_loader_registry_register_many_with_existing_extension() -> None:
    loader = LoaderRegistry()
    loader.register("json", JsonLoader())
    with pytest.raises(RuntimeError, match=r"Loader .* already registered"):
        loader.register_many({"json": JsonLoader(), "txt": TextLoader()})


def test_loader_registry_has_loader_true() -> None:
    assert LoaderRegistry({"txt": TextLoader()}).has_loader("txt")


def test_loader_registry_has_loader_false() -> None:
    assert not LoaderRegistry().has_loader("txt")


def test_loader_registry_find_loader_txt() -> None:
    assert isinstance(LoaderRegistry({"txt": TextLoader()}).find_loader("txt"), TextLoader)


def test_loader_registry_find_loader_incorrect_extension() -> None:
    with pytest.raises(ValueError, match=r"Incorrect extension:"):
        LoaderRegistry().find_loader("txt")


def test_loader_registry_registry_isolation() -> None:
    registry1 = LoaderRegistry({"json": JsonLoader()})
    registry2 = LoaderRegistry({"json": TextLoader()})

    assert objects_are_equal(registry1._registry, {"json": JsonLoader()})
    assert objects_are_equal(registry2._registry, {"json": TextLoader()})
