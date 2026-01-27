from __future__ import annotations

from unittest.mock import patch

import pytest

from iden.utils.imports import (
    check_yaml,
    is_yaml_available,
    raise_error_yaml_missing,
    yaml_available,
)


def my_function(n: int = 0) -> int:
    return 42 + n


def test_check_yaml_with_package() -> None:
    with patch("iden.utils.imports.yaml.is_yaml_available", lambda: True):
        check_yaml()


def test_check_yaml_without_package() -> None:
    with (
        patch("iden.utils.imports.yaml.is_yaml_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'yaml' package is required but not installed."),
    ):
        check_yaml()


def test_is_yaml_available() -> None:
    assert isinstance(is_yaml_available(), bool)


def test_yaml_available_with_package() -> None:
    with patch("iden.utils.imports.yaml.is_yaml_available", lambda: True):
        fn = yaml_available(my_function)
        assert fn(2) == 44


def test_yaml_available_without_package() -> None:
    with patch("iden.utils.imports.yaml.is_yaml_available", lambda: False):
        fn = yaml_available(my_function)
        assert fn(2) is None


def test_yaml_available_decorator_with_package() -> None:
    with patch("iden.utils.imports.yaml.is_yaml_available", lambda: True):

        @yaml_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_yaml_available_decorator_without_package() -> None:
    with patch("iden.utils.imports.yaml.is_yaml_available", lambda: False):

        @yaml_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_error_yaml_missing() -> None:
    with pytest.raises(RuntimeError, match=r"'yaml' package is required but not installed."):
        raise_error_yaml_missing()
