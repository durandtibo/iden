r"""Contain safetensors-based shard implementations."""

from __future__ import annotations

__all__ = ["TorchSafetensorsShard", "create_torch_safetensors_shard"]

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.utils import is_torch_available
from objectory import OBJECT_TARGET

from iden.constants import KWARGS, LOADER
from iden.io import JsonSaver
from iden.io.safetensors import TorchLoader, TorchSaver
from iden.shard.file import FileShard
from iden.utils.path import sanitize_path

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class TorchSafetensorsShard(FileShard[dict[str, torch.Tensor]]):
    r"""Implement a safetensors shard for ``torch.Tensor``s.

    The data are stored in a safetensors file.

    Args:
        uri: The shard's URI.
        path: Specifies the path to the safetensors file.

    Raises:
        RuntimeError: if ``safetensors`` or ``torch`` is not installed.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> from iden.shard import TorchSafetensorsShard
    >>> from iden.io.safetensors import TorchSaver
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     file = Path(tmpdir).joinpath("data.safetensors")
    ...     TorchSaver().save({"key1": torch.ones(2, 3), "key2": torch.arange(5)}, file)
    ...     shard = TorchSafetensorsShard(uri="file:///data/1234456789", path=file)
    ...     shard.get_data()
    ...
    {'key1': tensor([[1., 1., 1.], [1., 1., 1.]]), 'key2': tensor([0, 1, 2, 3, 4])}

    ```
    """

    def __init__(self, uri: str, path: Path | str) -> None:
        super().__init__(uri, path, loader=TorchLoader())

    @classmethod
    def generate_uri_config(cls, path: Path) -> dict:
        r"""Generate the minimal config that is used to load the shard
        from its URI.

        The config must be compatible with the JSON format.

        Args:
            path: The path to the pickle file.

        Returns:
            The minimal config to load the shard from its URI.

        Example usage:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from iden.shard import TorchSafetensorsShard
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     file = Path(tmpdir).joinpath("data.safetensors")
        ...     TorchSafetensorsShard.generate_uri_config(file)
        ...
        {'kwargs': {'path': '.../data.safetensors'},
         'loader': {'_target_': 'iden.shard.loader.TorchSafetensorsShardLoader'}}

        ```
        """
        return {
            KWARGS: {"path": sanitize_path(path).as_posix()},
            LOADER: {OBJECT_TARGET: "iden.shard.loader.TorchSafetensorsShardLoader"},
        }


def create_torch_safetensors_shard(
    data: dict[str, torch.Tensor], uri: str, path: Path | None = None
) -> TorchSafetensorsShard:
    r"""Create a ``TorchSafetensorsShard`` from data.

    Note:
        It is a utility function to create a ``TorchSafetensorsShard``
            from its data and URI. It is possible to create a
            ``TorchSafetensorsShard`` in other ways.

    Args:
        data: The data to save in the safetensors file.
        uri: The shard's URI.
        path: The path to the safetensors file. If ``None``, a path is
            automatically based on the URI.

    Returns:
        The ``TorchSafetensorsShard`` object.

    Raises:
        RuntimeError: if ``safetensors`` or ``torch`` is not installed.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> import torch
    >>> from iden.shard import create_torch_safetensors_shard
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     shard = create_torch_safetensors_shard(
    ...         data={"key1": torch.ones(2, 3), "key2": torch.arange(5)},
    ...         uri=Path(tmpdir).joinpath("my_uri").as_uri()
    ...     )
    ...     shard.get_data()
    ...
    {'key1': tensor([[1., 1., 1.], [1., 1., 1.]]), 'key2': tensor([0, 1, 2, 3, 4])}

    ```
    """
    if path is None:
        path = sanitize_path(uri + ".safetensors")
    logger.info(f"Saving URI file {uri}")
    JsonSaver().save(TorchSafetensorsShard.generate_uri_config(path), sanitize_path(uri))
    logger.info(f"Saving data in file {path}")
    TorchSaver().save(data, path)
    return TorchSafetensorsShard(uri, path)
