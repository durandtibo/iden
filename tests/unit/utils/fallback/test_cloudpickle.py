from __future__ import annotations

from types import ModuleType

import pytest

from iden.utils.fallback.cloudpickle import cloudpickle


def test_cloudpickle_is_module_type() -> None:
    assert isinstance(cloudpickle, ModuleType)


def test_cloudpickle_module_name() -> None:
    assert cloudpickle.__name__ == "cloudpickle"


def test_cloudpickle_dump_call() -> None:
    with pytest.raises(RuntimeError, match=r"'cloudpickle' package is required but not installed."):
        cloudpickle.dump()


def test_cloudpickle_load_call() -> None:
    with pytest.raises(RuntimeError, match=r"'cloudpickle' package is required but not installed."):
        cloudpickle.load()
