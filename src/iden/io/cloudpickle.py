r"""Contain cloudpickle-based data loaders and savers."""

from __future__ import annotations

__all__ = [
    "CloudpickleLoader",
    "CloudpickleSaver",
    "load_cloudpickle",
    "save_cloudpickle",
    "get_loader_mapping",
]

from pathlib import Path
from typing import Any
from unittest.mock import Mock

from iden.io.base import BaseFileSaver, BaseLoader
from iden.utils.imports import check_cloudpickle, is_cloudpickle_available

if is_cloudpickle_available():
    import cloudpickle
else:  # pragma: no cover
    cloudpickle = Mock()


class CloudpickleLoader(BaseLoader[Any]):
    r"""Implement a data loader to load data in a pickle file with
    cloudpickle.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.io import save_cloudpickle, CloudpickleLoader
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pt")
    ...     save_cloudpickle({"key1": [1, 2, 3], "key2": "abc"}, path)
    ...     data = CloudpickleLoader().load(path)
    ...     data
    ...
    {'key1': [1, 2, 3], 'key2': 'abc'}

    ```
    """

    def __init__(self) -> None:
        check_cloudpickle()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, path: Path) -> Any:
        with Path.open(path, mode="rb") as file:
            return cloudpickle.load(file)  # noqa: S301


class CloudpickleSaver(BaseFileSaver[Any]):
    r"""Implement a file saver to save data with a pickle file with
    cloudpickle.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.io import CloudpickleSaver, CloudpickleLoader
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pt")
    ...     CloudpickleSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
    ...     data = CloudpickleLoader().load(path)
    ...     data
    ...
    {'key1': [1, 2, 3], 'key2': 'abc'}

    ```
    """

    def __init__(self) -> None:
        check_cloudpickle()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _save_file(self, to_save: Any, path: Path) -> None:
        with Path.open(path, mode="wb") as file:
            cloudpickle.dump(to_save, file)


def load_cloudpickle(path: Path) -> Any:
    r"""Load the data from a given pickle file with cloudpickle.

    Args:
        path: The path to the pickle file.

    Returns:
        The data from the pickle file.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.io import save_cloudpickle, load_cloudpickle
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pt")
    ...     save_cloudpickle({"key1": [1, 2, 3], "key2": "abc"}, path)
    ...     data = load_cloudpickle(path)
    ...     data
    ...
    {'key1': [1, 2, 3], 'key2': 'abc'}

    ```
    """
    return CloudpickleLoader().load(path)


def save_cloudpickle(to_save: Any, path: Path, *, exist_ok: bool = False) -> None:
    r"""Save the given data in a pickle file with cloudpickle.

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
    >>> from iden.io import save_cloudpickle, load_cloudpickle
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pt")
    ...     save_cloudpickle({"key1": [1, 2, 3], "key2": "abc"}, path)
    ...     data = load_cloudpickle(path)
    ...     data
    ...
    {'key1': [1, 2, 3], 'key2': 'abc'}

    ```
    """
    CloudpickleSaver().save(to_save, path, exist_ok=exist_ok)


def get_loader_mapping() -> dict[str, BaseLoader]:
    r"""Get a default mapping between the file extensions and loaders.

    Returns:
        The mapping between the file extensions and loaders.

    Example usage:

    ```pycon

    >>> from iden.io.cloudpickle import get_loader_mapping
    >>> get_loader_mapping()
    {}

    ```
    """
    return {}