r"""Contain numpy-based data loaders and savers."""

from __future__ import annotations

__all__ = ["NumpyZLoader", "NumpyZSaver"]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar
from unittest.mock import Mock

from coola.utils import check_numpy, is_numpy_available

from iden.io.base import BaseFileSaver, BaseLoader

if TYPE_CHECKING:
    from pathlib import Path

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

T = TypeVar("T", Sequence, Mapping)


class NumpyZLoader(BaseLoader[T]):
    r"""Implement a data loader to load data in a pickle file.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.io import save_pickle, NumpyZLoader
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pkl")
    ...     save_pickle({"key1": [1, 2, 3], "key2": "abc"}, path)
    ...     data = NumpyZLoader().load(path)
    ...     data
    ...
    {'key1': [1, 2, 3], 'key2': 'abc'}

    ```
    """

    def __init__(self) -> None:
        check_numpy()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, path: Path) -> T:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}


class NumpyZSaver(BaseFileSaver[T]):
    r"""Implement a file saver to save data with a pickle file.

    Args:
        protocol: The pickle protocol. By default, it uses the
            highest protocol available.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.io import NumpyZSaver, NumpyZLoader
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pkl")
    ...     NumpyZSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
    ...     data = NumpyZLoader().load(path)
    ...     data
    ...
    {'key1': [1, 2, 3], 'key2': 'abc'}

    ```
    """

    def __init__(self) -> None:
        check_numpy()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _save_file(self, to_save: T, path: Path) -> None:
        # Save to tmp, then commit by moving the file in case the job gets
        # interrupted while writing the file
        tmp_path = path.parents[0].joinpath(f"{path.name}.tmp.npz")
        if isinstance(to_save, Mapping):
            np.savez(tmp_path, **to_save)
        elif isinstance(to_save, Sequence):
            np.savez(tmp_path, *to_save)
        else:
            msg = f"Incorrect type: {type(to_save)}"
            raise TypeError(msg)
        tmp_path.rename(path)
