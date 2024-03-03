r"""Contain in-memory shard implementations."""

from __future__ import annotations

__all__ = ["PickleShard", "save_uri_file"]

from typing import TYPE_CHECKING, Any, TypeVar

from iden.shard.base import BaseShard
from iden.utils.io import load_json, load_pickle, save_json
from iden.utils.path import sanitize_path

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


class PickleShard(BaseShard[T]):
    r"""Implement a pickle shard.

    The data are stored in a pickle file.

    Args:
        uri: The URI associated to the shard.
        path: Specifies the path to the pickle file.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard import PickleShard
    >>> from iden.utils.io import save_pickle
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pkl")
    ...     save_pickle([1, 2, 3], path)
    ...     shard = PickleShard(uri="file:///data/1234456789", path=path)
    ...     shard.get_data()
    ...
    [1, 2, 3]

    ```
    """

    def __init__(self, uri: str, path: Path | str) -> None:
        self._uri = uri
        self._path = sanitize_path(path)
        self._data = None

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(uri={self.get_uri()})"

    @property
    def path(self) -> Path:
        r"""The path to the pickle file."""
        return self._path

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.get_uri() == other.get_uri() and self.path == other.path

    def get_data(self) -> T:
        if self._data is None:
            self._data = load_pickle(self._path)
        return self._data

    def get_uri(self) -> str:
        return self._uri

    @classmethod
    def from_uri(cls, uri: str) -> PickleShard:
        config = load_json(sanitize_path(uri))
        return cls(uri=uri, path=config["path"])


def save_uri_file(uri: str, path: Path | str) -> None:
    r"""Save the Uniform Resource Identifier (URI) file.

    Args:
        uri: The URI associated to the shard.
        path: Specifies the path to the pickle file.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard.pickle_ import save_uri_file
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("data.pkl")
    ...     save_uri_file(uri="file:///data/1234456789", path=path)  # xdoctest: +SKIP()
    ...

    ```
    """
    save_json({"path": sanitize_path(path).as_posix()}, sanitize_path(uri))
