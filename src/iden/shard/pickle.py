r"""Contain pickle-based shard implementations."""

from __future__ import annotations

__all__ = ["PickleShard", "create_pickle_shard", "save_uri_file"]

from typing import TYPE_CHECKING, TypeVar

from objectory import OBJECT_TARGET

from iden.constants import KWARGS, LOADER
from iden.io import PickleLoader, save_json, save_pickle
from iden.shard.file import FileShard
from iden.utils.path import sanitize_path

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


class PickleShard(FileShard[T]):
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
    >>> from iden.io import save_pickle
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     file = Path(tmpdir).joinpath("data.pkl")
    ...     save_pickle([1, 2, 3], file)
    ...     shard = PickleShard(uri="file:///data/1234456789", path=file)
    ...     shard.get_data()
    ...
    [1, 2, 3]

    ```
    """

    def __init__(self, uri: str, path: Path | str) -> None:
        super().__init__(uri, path, loader=PickleLoader())


def create_pickle_shard(data: T, uri: str) -> PickleShard:
    r"""Create a ``PickleShard`` from data.

    Note:
        It is a utility function to create a ``PickleShard`` from its
            data and URI. It is possible to create a ``PickleShard``
            in other ways.

    Args:
        data: The data to save in the pickle file.
        uri: The URI associated to the shard.

    Returns:
        The ``PickleShard`` object.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard.pickle import create_pickle_shard
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     shard = create_pickle_shard([1, 2, 3], uri=Path(tmpdir).joinpath("my_uri").as_uri())
    ...     shard.get_data()
    ...
    [1, 2, 3]

    ```
    """
    path = sanitize_path(uri + ".pkl")
    save_pickle(data, path)
    save_uri_file(uri, path)
    return PickleShard(uri, path)


def save_uri_file(uri: str, path: Path | str) -> None:
    r"""Save the Uniform Resource Identifier (URI) file for a
    ``PickleShard``.

    Args:
        uri: The URI associated to the shard.
        path: Specifies the path to the pickle file.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard.pickle import save_uri_file
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     file = Path(tmpdir).joinpath("data.pkl")
    ...     save_uri_file(uri="file:///data/my_uri", path=file)  # xdoctest: +SKIP()
    ...

    ```
    """
    config = {
        KWARGS: {"path": sanitize_path(path).as_posix()},
        LOADER: {OBJECT_TARGET: "iden.shard.loader.PickleShardLoader"},
    }
    save_json(config, sanitize_path(uri))
