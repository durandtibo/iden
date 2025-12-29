from __future__ import annotations

from types import ModuleType

import pytest

from iden.utils.fallback.joblib import joblib


def test_joblib_is_module_type() -> None:
    assert isinstance(joblib, ModuleType)


def test_joblib_module_name() -> None:
    assert joblib.__name__ == "joblib"


def test_joblib_dump_call() -> None:
    with pytest.raises(RuntimeError, match=r"'joblib' package is required but not installed."):
        joblib.dump()


def test_joblib_load_call() -> None:
    with pytest.raises(RuntimeError, match=r"'joblib' package is required but not installed."):
        joblib.load()
