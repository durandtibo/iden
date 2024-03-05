r"""Contain text-based data loaders and savers."""

from __future__ import annotations

__all__ = ["TextLoader", "TextSaver"]

from typing import TYPE_CHECKING, Any, TypeVar

from iden.io.base import BaseFileSaver, BaseLoader
from iden.utils.io import load_text, save_text

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


class TextLoader(BaseLoader[Any]):
    r"""Implement a data loader to load data in a text file."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, path: Path) -> Any:
        return load_text(path)


class TextSaver(BaseFileSaver[Any]):
    r"""Implement a file saver to save data with a text file."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _save_file(self, to_save: Any, path: Path) -> None:
        save_text(to_save, path)
