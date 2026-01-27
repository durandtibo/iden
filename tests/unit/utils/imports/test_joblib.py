from __future__ import annotations

from unittest.mock import patch

import pytest

from iden.utils.imports import (
    check_joblib,
    is_joblib_available,
    joblib_available,
    raise_error_joblib_missing,
)


def my_function(n: int = 0) -> int:
    return 42 + n


def test_check_joblib_with_package() -> None:
    with patch("iden.utils.imports.joblib.is_joblib_available", lambda: True):
        check_joblib()


def test_check_joblib_without_package() -> None:
    with (
        patch("iden.utils.imports.joblib.is_joblib_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'joblib' package is required but not installed."),
    ):
        check_joblib()


def test_is_joblib_available() -> None:
    assert isinstance(is_joblib_available(), bool)


def test_joblib_available_with_package() -> None:
    with patch("iden.utils.imports.joblib.is_joblib_available", lambda: True):
        fn = joblib_available(my_function)
        assert fn(2) == 44


def test_joblib_available_without_package() -> None:
    with patch("iden.utils.imports.joblib.is_joblib_available", lambda: False):
        fn = joblib_available(my_function)
        assert fn(2) is None


def test_joblib_available_decorator_with_package() -> None:
    with patch("iden.utils.imports.joblib.is_joblib_available", lambda: True):

        @joblib_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_joblib_available_decorator_without_package() -> None:
    with patch("iden.utils.imports.joblib.is_joblib_available", lambda: False):

        @joblib_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_error_joblib_missing() -> None:
    with pytest.raises(RuntimeError, match=r"'joblib' package is required but not installed."):
        raise_error_joblib_missing()
