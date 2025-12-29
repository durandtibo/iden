from __future__ import annotations

from types import ModuleType

import pytest

from iden.utils.fallback.yaml import yaml


def test_yaml_is_module_type() -> None:
    assert isinstance(yaml, ModuleType)


def test_yaml_module_name() -> None:
    assert yaml.__name__ == "yaml"


def test_yaml_dump_call() -> None:
    with pytest.raises(RuntimeError, match=r"'yaml' package is required but not installed."):
        yaml.dump()


def test_yaml_safe_load_call() -> None:
    with pytest.raises(RuntimeError, match=r"'yaml' package is required but not installed."):
        yaml.safe_load()
