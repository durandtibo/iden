r"""Contain joblib-based data loaders and savers."""

from __future__ import annotations

__all__ = [
    "JoblibLoader",
    "JoblibSaver",
    "load_joblib",
    "save_joblib",
    "get_loader_mapping",
]

from pathlib import Path
from typing import Any
from unittest.mock import Mock

from iden.io.base import BaseFileSaver, BaseLoader
from iden.utils.imports import check_joblib, is_joblib_available

if is_joblib_available():
    import joblib
else:  # pragma: no cover
    joblib = Mock()


class JoblibLoader(BaseLoader[Any]):
    r"""Implement a data loader to load data in a pickle file with
    joblib.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.io import save_joblib, JoblibLoader
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pt")
    ...     save_joblib({"key1": [1, 2, 3], "key2": "abc"}, path)
    ...     data = JoblibLoader().load(path)
    ...     data
    ...
    {'key1': [1, 2, 3], 'key2': 'abc'}

    ```
    """

    def __init__(self) -> None:
        check_joblib()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, path: Path) -> Any:
        with Path.open(path, mode="rb") as file:
            return joblib.load(file)  # noqa: S301


class JoblibSaver(BaseFileSaver[Any]):
    r"""Implement a file saver to save data with a pickle file with
    joblib.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.io import JoblibSaver, JoblibLoader
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pt")
    ...     JoblibSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
    ...     data = JoblibLoader().load(path)
    ...     data
    ...
    {'key1': [1, 2, 3], 'key2': 'abc'}

    ```
    """

    def __init__(self) -> None:
        check_joblib()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _save_file(self, to_save: Any, path: Path) -> None:
        with Path.open(path, mode="wb") as file:
            joblib.dump(to_save, file)


def load_joblib(path: Path) -> Any:
    r"""Load the data from a given pickle file with joblib.

    Args:
        path: The path to the pickle file.

    Returns:
        The data from the pickle file.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.io import save_joblib, load_joblib
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pt")
    ...     save_joblib({"key1": [1, 2, 3], "key2": "abc"}, path)
    ...     data = load_joblib(path)
    ...     data
    ...
    {'key1': [1, 2, 3], 'key2': 'abc'}

    ```
    """
    return JoblibLoader().load(path)


def save_joblib(to_save: Any, path: Path, *, exist_ok: bool = False) -> None:
    r"""Save the given data in a pickle file with joblib.

    Args:
        to_save: The data to write in a pickle file.
        path: The path where to write the pickle file.
        exist_ok: If ``exist_ok`` is ``False`` (the default),
            ``FileExistsError`` is raised if the target file
            already exists. If ``exist_ok`` is ``True``,
            ``FileExistsError`` will not be raised unless the
            given path already exists in the file system and is
            not a file.

    Raises:
        FileExistsError: if the file already exists.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.io import save_joblib, load_joblib
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pt")
    ...     save_joblib({"key1": [1, 2, 3], "key2": "abc"}, path)
    ...     data = load_joblib(path)
    ...     data
    ...
    {'key1': [1, 2, 3], 'key2': 'abc'}

    ```
    """
    JoblibSaver().save(to_save, path, exist_ok=exist_ok)


def get_loader_mapping() -> dict[str, BaseLoader]:
    r"""Get a default mapping between the file extensions and loaders.

    Returns:
        The mapping between the file extensions and loaders.

    Example usage:

    ```pycon

    >>> from iden.io.joblib import get_loader_mapping
    >>> get_loader_mapping()
    {}

    ```
    """
    return {}