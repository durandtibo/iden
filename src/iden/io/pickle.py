r"""Contain pickle-based data loaders and savers."""

from __future__ import annotations

__all__ = ["PickleLoader", "PickleSaver"]

import pickle
from typing import TYPE_CHECKING, Any, TypeVar

from iden.io.base import BaseFileSaver, BaseLoader
from iden.utils.io import load_pickle, save_pickle

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


class PickleLoader(BaseLoader[Any]):
    r"""Implement a data loader to load data in a pickle file."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, path: Path) -> Any:
        return load_pickle(path)


class PickleSaver(BaseFileSaver[Any]):
    r"""Implement a file saver to save data with a pickle file.

    Args:
        protocol: The pickle protocol. By default, it uses the
            highest protocol available.
    """

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
        self._protocol = protocol

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(protocol={self._protocol})"

    def _save_file(self, to_save: Any, path: Path) -> None:
        save_pickle(to_save, path, protocol=self._protocol)
