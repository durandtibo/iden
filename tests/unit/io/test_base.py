from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coola.equality.testers import EqualityTester
from objectory import OBJECT_TARGET

from iden.io import (
    BaseLoader,
    BaseSaver,
    JsonLoader,
    JsonSaver,
    is_loader_config,
    is_saver_config,
    setup_loader,
    setup_saver,
)

if TYPE_CHECKING:
    import pytest

######################################
#     Tests for is_loader_config     #
######################################


def test_is_loader_config_true() -> None:
    assert is_loader_config({OBJECT_TARGET: "iden.io.JsonLoader"})


def test_is_loader_config_false() -> None:
    assert not is_loader_config({OBJECT_TARGET: "iden.io.JsonSaver"})


#####################################
#     Tests for is_saver_config     #
#####################################


def test_is_saver_config_true() -> None:
    assert is_saver_config({OBJECT_TARGET: "iden.io.JsonSaver"})


def test_is_saver_config_false() -> None:
    assert not is_saver_config({OBJECT_TARGET: "iden.io.JsonLoader"})


##################################
#     Tests for setup_loader     #
##################################


def test_setup_loader_object() -> None:
    loader = JsonLoader()
    assert setup_loader(loader) is loader


def test_setup_loader_dict() -> None:
    assert isinstance(setup_loader({OBJECT_TARGET: "iden.io.JsonLoader"}), JsonLoader)


def test_setup_loader_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_loader({OBJECT_TARGET: "iden.io.JsonSaver"}), JsonSaver)
        assert caplog.messages


#################################
#     Tests for setup_saver     #
#################################


def test_setup_saver_object() -> None:
    saver = JsonSaver()
    assert setup_saver(saver) is saver


def test_setup_saver_dict() -> None:
    assert isinstance(setup_saver({OBJECT_TARGET: "iden.io.JsonSaver"}), JsonSaver)


def test_setup_saver_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_saver({OBJECT_TARGET: "iden.io.JsonLoader"}), JsonLoader)
        assert caplog.messages


def test_equality_tester_has_comparator() -> None:
    assert EqualityTester.has_comparator(BaseLoader)
    assert EqualityTester.has_comparator(BaseSaver)
