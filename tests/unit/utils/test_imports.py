from unittest.mock import patch

import pytest

from iden.utils.imports import (
    check_safetensors,
    check_yaml,
    is_safetensors_available,
    is_yaml_available,
    safetensors_available,
    yaml_available,
)


def my_function(n: int = 0) -> int:
    return 42 + n


#######################
#     safetensors     #
#######################


def test_check_safetensors_with_package() -> None:
    with patch("iden.utils.imports.is_safetensors_available", lambda: True):
        check_safetensors()


def test_check_safetensors_without_package() -> None:
    with (
        patch("iden.utils.imports.is_safetensors_available", lambda: False),
        pytest.raises(RuntimeError, match="`safetensors` package is required but not installed."),
    ):
        check_safetensors()


def test_is_safetensors_available() -> None:
    assert isinstance(is_safetensors_available(), bool)


def test_safetensors_available_with_package() -> None:
    with patch("iden.utils.imports.is_safetensors_available", lambda: True):
        fn = safetensors_available(my_function)
        assert fn(2) == 44


def test_safetensors_available_without_package() -> None:
    with patch("iden.utils.imports.is_safetensors_available", lambda: False):
        fn = safetensors_available(my_function)
        assert fn(2) is None


def test_safetensors_available_decorator_with_package() -> None:
    with patch("iden.utils.imports.is_safetensors_available", lambda: True):

        @safetensors_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_safetensors_available_decorator_without_package() -> None:
    with patch("iden.utils.imports.is_safetensors_available", lambda: False):

        @safetensors_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


################
#     yaml     #
################


def test_check_yaml_with_package() -> None:
    with patch("iden.utils.imports.is_yaml_available", lambda: True):
        check_yaml()


def test_check_yaml_without_package() -> None:
    with (
        patch("iden.utils.imports.is_yaml_available", lambda: False),
        pytest.raises(RuntimeError, match="`yaml` package is required but not installed."),
    ):
        check_yaml()


def test_is_yaml_available() -> None:
    assert isinstance(is_yaml_available(), bool)


def test_yaml_available_with_package() -> None:
    with patch("iden.utils.imports.is_yaml_available", lambda: True):
        fn = yaml_available(my_function)
        assert fn(2) == 44


def test_yaml_available_without_package() -> None:
    with patch("iden.utils.imports.is_yaml_available", lambda: False):
        fn = yaml_available(my_function)
        assert fn(2) is None


def test_yaml_available_decorator_with_package() -> None:
    with patch("iden.utils.imports.is_yaml_available", lambda: True):

        @yaml_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_yaml_available_decorator_without_package() -> None:
    with patch("iden.utils.imports.is_yaml_available", lambda: False):

        @yaml_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None
