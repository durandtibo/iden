r"""Contain in-memory shard implementations."""

from __future__ import annotations

__all__ = ["PickleShard"]

from typing import TYPE_CHECKING, TypeVar

from iden.shard.base import BaseShard
from iden.utils.io import load_pickle
from iden.utils.path import sanitize_path

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


class PickleShard(BaseShard[T]):
    r"""Implement a pickle shard.

    The data are stored in a pickle file.

    Args:
        path: Specifies the path where to store the pickle file.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard import PickleShard
    >>> from iden.utils.io import save_pickle
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pkl")
    ...     save_pickle([1, 2, 3], path)
    ...     shard = PickleShard(path)
    ...     shard.get_data()
    ...
    [1, 2, 3]

    ```
    """

    def __init__(self, path: Path | str) -> None:
        self._path = sanitize_path(path)
        self._data = None

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(uri={self.get_uri()})"

    def get_data(self) -> T:
        if self._data is None:
            self._data = load_pickle(self._path)
        return self._data

    def get_uri(self) -> str:
        return self._path.as_uri()
