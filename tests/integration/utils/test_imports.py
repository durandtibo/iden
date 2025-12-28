from __future__ import annotations

import pytest

from iden.testing import (
    cloudpickle_available,
    cloudpickle_not_available,
    joblib_available,
    joblib_not_available,
    safetensors_available,
    safetensors_not_available,
    yaml_available,
    yaml_not_available,
)
from iden.utils.imports import (
    check_cloudpickle,
    check_joblib,
    check_safetensors,
    check_yaml,
    is_cloudpickle_available,
    is_joblib_available,
    is_safetensors_available,
    is_yaml_available,
)

#######################
#     cloudpickle     #
#######################


@cloudpickle_available
def test_check_cloudpickle_with_package() -> None:
    check_cloudpickle()


@cloudpickle_not_available
def test_check_cloudpickle_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'cloudpickle' package is required but not installed."):
        check_cloudpickle()


@cloudpickle_available
def test_is_cloudpickle_available_true() -> None:
    assert is_cloudpickle_available()


@cloudpickle_not_available
def test_is_cloudpickle_available_false() -> None:
    assert not is_cloudpickle_available()


##################
#     joblib     #
##################


@joblib_available
def test_check_joblib_with_package() -> None:
    check_joblib()


@joblib_not_available
def test_check_joblib_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'joblib' package is required but not installed."):
        check_joblib()


@joblib_available
def test_is_joblib_available_true() -> None:
    assert is_joblib_available()


@joblib_not_available
def test_is_joblib_available_false() -> None:
    assert not is_joblib_available()


#######################
#     safetensors     #
#######################


@safetensors_available
def test_check_safetensors_with_package() -> None:
    check_safetensors()


@safetensors_not_available
def test_check_safetensors_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'safetensors' package is required but not installed."):
        check_safetensors()


@safetensors_available
def test_is_safetensors_available_true() -> None:
    assert is_safetensors_available()


@safetensors_not_available
def test_is_safetensors_available_false() -> None:
    assert not is_safetensors_available()


################
#     yaml     #
################


@yaml_available
def test_check_yaml_with_package() -> None:
    check_yaml()


@yaml_not_available
def test_check_yaml_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'yaml' package is required but not installed."):
        check_yaml()


@yaml_available
def test_is_yaml_available_true() -> None:
    assert is_yaml_available()


@yaml_not_available
def test_is_yaml_available_false() -> None:
    assert not is_yaml_available()
