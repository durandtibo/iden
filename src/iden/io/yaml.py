r"""Contain YAML-based data loaders and savers."""

from __future__ import annotations

__all__ = ["YamlLoader", "YamlSaver", "load_yaml", "save_yaml", "get_loader_mapping"]

from pathlib import Path
from typing import Any, TypeVar

import yaml

from iden.io.base import BaseFileSaver, BaseLoader

T = TypeVar("T")


class YamlLoader(BaseLoader[Any]):
    r"""Implement a data loader to load data in a YAML file.

    Example usage:

    ```pycon
    >>> from pathlib import Path
    >>> from iden.io import load_yaml
    >>> data = YamlLoader().load(Path("/path/to/data.yaml"))  # xdoctest: +SKIP()

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, path: Path) -> Any:
        with Path.open(path, mode="rb") as file:
            return yaml.safe_load(file)


class YamlSaver(BaseFileSaver[Any]):
    r"""Implement a file saver to save data with a YAML file.

    Example usage:

    ```pycon
    >>> from pathlib import Path
    >>> from iden.io import YamlSaver
    >>> YamlSaver().save({"key": "value"}, Path("/path/to/data.yaml"))  # xdoctest: +SKIP()

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _save_file(self, to_save: Any, path: Path) -> None:
        # Save to tmp, then commit by moving the file in case the job gets
        # interrupted while writing the file
        tmp_path = path.parents[0].joinpath(f"{path.name}.tmp")
        with Path.open(tmp_path, mode="w") as file:
            yaml.dump(to_save, file, Dumper=yaml.Dumper)
        tmp_path.rename(path)


def load_yaml(path: Path) -> Any:
    r"""Load the data from a given YAML file.

    Args:
        path: Specifies the path to the YAML file.

    Returns:
        The data from the YAML file.

    Example usage:

    ```pycon
    >>> from pathlib import Path
    >>> from iden.io import load_yaml
    >>> data = load_yaml(Path("/path/to/data.yaml"))  # xdoctest: +SKIP()

    ```
    """
    return YamlLoader().load(path)


def save_yaml(to_save: Any, path: Path, *, exist_ok: bool = False) -> None:
    r"""Save the given data in a YAML file.

    Args:
        to_save: Specifies the data to write in a YAML file.
        path: Specifies the path where to write the YAML file.
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
    >>> from pathlib import Path
    >>> from iden.io import save_yaml
    >>> save_yaml({"key": "value"}, Path("/path/to/data.yaml"))  # xdoctest: +SKIP()

    ```
    """
    YamlSaver().save(to_save, path, exist_ok=exist_ok)


def get_loader_mapping() -> dict[str, BaseLoader]:
    r"""Get a default mapping between the file extensions and loaders.

    Returns:
        The mapping between the file extensions and loaders.

    Example usage:

    ```pycon
    >>> from iden.io.yaml import get_loader_mapping
    >>> get_loader_mapping()
    {'yaml': YamlLoader(), 'yml': YamlLoader()}

    ```
    """
    loader = YamlLoader()
    return {"yaml": loader, "yml": loader}
