r"""Contain the base class to implement a data loader or saver
object."""

from __future__ import annotations

__all__ = ["BaseSaver", "BaseFileSaver"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


class BaseSaver(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a data saver."""

    @abstractmethod
    def save(self, to_save: T, path: Path, exist_ok: bool = False) -> None:
        r"""Save the data into the given path.

        Args:
            to_save: The data to save. The data should be compatible
                with the saving engine.
            path: Specifies the path where to save the data.
            exist_ok: If ``exist_ok`` is ``False`` (the default),
                an exception is raised if the target path already
                exists.
        """


class BaseFileSaver(BaseSaver[T]):
    r"""Define the base class to implement a file saver."""

    def save(self, to_save: T, path: Path, exist_ok: bool = False) -> None:
        r"""Save the data into the given path.

        Args:
            to_save: The data to save. The data should be compatible
                with the saving engine.
            path: Specifies the path where to save the data.
            exist_ok: If ``exist_ok`` is ``False`` (the default),
                ``FileExistsError`` is raised if the target file
                already exists. If ``exist_ok`` is ``True``,
                ``FileExistsError`` will not be raised unless the
                given path already exists in the file system and is
                not a file.

        Raises:
            FileExistsError: if the file already exists.
        """
        if path.is_dir():
            msg = f"path ({path}) is a directory"
            raise IsADirectoryError(msg)
        if path.is_file() and not exist_ok:
            msg = (
                f"path ({path}) already exists. "
                f"Please use `exist_ok=True` if you want to overwrite the setter for this name"
            )
            raise FileExistsError(msg)
        self._save_file(to_save, path)

    @abstractmethod
    def _save_file(self, to_save: T, path: Path) -> None:
        r"""Save the data into the given file.

        Args:
            to_save: The data to save. The data should be compatible
                with the saving engine.
            path: Specifies the path where to save the data.
        """
