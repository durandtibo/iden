from __future__ import annotations

from unittest.mock import patch

import pytest

from iden.utils.imports import (
    check_cloudpickle,
    cloudpickle_available,
    is_cloudpickle_available,
    raise_error_cloudpickle_missing,
)


def my_function(n: int = 0) -> int:
    return 42 + n


def test_check_cloudpickle_with_package() -> None:
    with patch("iden.utils.imports.cloudpickle.is_cloudpickle_available", lambda: True):
        check_cloudpickle()


def test_check_cloudpickle_without_package() -> None:
    with (
        patch("iden.utils.imports.cloudpickle.is_cloudpickle_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'cloudpickle' package is required but not installed."),
    ):
        check_cloudpickle()


def test_is_cloudpickle_available() -> None:
    assert isinstance(is_cloudpickle_available(), bool)


def test_cloudpickle_available_with_package() -> None:
    with patch("iden.utils.imports.cloudpickle.is_cloudpickle_available", lambda: True):
        fn = cloudpickle_available(my_function)
        assert fn(2) == 44


def test_cloudpickle_available_without_package() -> None:
    with patch("iden.utils.imports.cloudpickle.is_cloudpickle_available", lambda: False):
        fn = cloudpickle_available(my_function)
        assert fn(2) is None


def test_cloudpickle_available_decorator_with_package() -> None:
    with patch("iden.utils.imports.cloudpickle.is_cloudpickle_available", lambda: True):

        @cloudpickle_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_cloudpickle_available_decorator_without_package() -> None:
    with patch("iden.utils.imports.cloudpickle.is_cloudpickle_available", lambda: False):

        @cloudpickle_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_error_cloudpickle_missing() -> None:
    with pytest.raises(RuntimeError, match=r"'cloudpickle' package is required but not installed."):
        raise_error_cloudpickle_missing()
