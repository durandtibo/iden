from __future__ import annotations

__all__ = ["FileShard"]

from typing import TYPE_CHECKING, Any, TypeVar

from iden.constants import KWARGS
from iden.io import AutoFileLoader, BaseLoader, load_json, setup_loader
from iden.shard.base import BaseShard
from iden.utils.path import sanitize_path

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


class FileShard(BaseShard[T]):
    r"""Implement a generic shard where the data are stored in a single
    file.

    Args:
        uri: The URI associated to the shard.
        path: Specifies the path to the pickle file.
        loader: The data loader or its configuration.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard import FileShard
    >>> from iden.io import save_json, JsonLoader
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     file = Path(tmpdir).joinpath("data.json")
    ...     save_json([1, 2, 3], file)
    ...     shard = FileShard(uri="file:///data/1234456789", path=file, loader=JsonLoader())
    ...     shard.get_data()
    ...
    [1, 2, 3]

    ```
    """

    def __init__(
        self, uri: str, path: Path | str, loader: BaseLoader[T] | dict | None = None
    ) -> None:
        self._uri = uri
        self._path = sanitize_path(path)
        self._loader = setup_loader(loader or AutoFileLoader())

        self._is_initialized = False
        self._data = None

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(uri={self.get_uri()})"

    @property
    def path(self) -> Path:
        r"""The path to the file with data."""
        return self._path

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.get_uri() == other.get_uri() and self.path == other.path

    def get_data(self) -> T:
        if not self._is_initialized:
            self._data = self._loader.load(self._path)
            self._is_initialized = True
        return self._data

    def get_uri(self) -> str:
        return self._uri

    @classmethod
    def from_uri(cls, uri: str) -> FileShard:
        config = load_json(sanitize_path(uri))
        return cls(uri=uri, **config[KWARGS])
