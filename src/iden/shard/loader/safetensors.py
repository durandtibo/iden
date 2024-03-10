r"""Contain safetensors shard loader implementations."""

from __future__ import annotations

__all__ = ["TorchSafetensorsShardLoader"]

from unittest.mock import Mock

from coola.utils import check_torch, is_torch_available

from iden.shard.loader.base import BaseShardLoader
from iden.shard.safetensors import TorchSafetensorsShard
from iden.utils.imports import check_safetensors

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


class TorchSafetensorsShardLoader(BaseShardLoader[dict[str, torch.Tensor]]):
    r"""Implement a safetensors shard loader for ``torch.Tensor``s.

    Raises:
        RuntimeError: if ``safetensors`` or ``torch`` is not installed.

    Example usage:

    ```pycon
    >>> import tempfile
    >>> import torch
    >>> from pathlib import Path
    >>> from iden.shard import create_torch_safetensors_shard
    >>> from iden.shard.loader import TorchSafetensorsShardLoader
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     uri = Path(tmpdir).joinpath("my_uri").as_uri()
    ...     _ = create_torch_safetensors_shard(
    ...         {"key1": torch.ones(2, 3), "key2": torch.arange(5)}, uri=uri
    ...     )
    ...     loader = TorchSafetensorsShardLoader()
    ...     shard = loader.load(uri)
    ...     shard
    ...
    TorchSafetensorsShard(uri=file:///.../my_uri)

    ```
    """

    def __init__(self) -> None:
        check_safetensors()
        check_torch()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load(self, uri: str) -> TorchSafetensorsShard:
        return TorchSafetensorsShard.from_uri(uri)
