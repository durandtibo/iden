r"""Contain I/O utility functions."""

from __future__ import annotations

__all__ = ["get_loader_mapping"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iden.io import BaseLoader


def get_loader_mapping() -> dict[str, BaseLoader]:
    r"""Get a default mapping between the file extensions and loaders.

    Returns:
        The mapping between the file extensions and loaders.

    Example usage:

    ```pycon
    >>> from iden.io.utils import get_loader_mapping
    >>> get_loader_mapping()
    {'json': JsonLoader(), 'pkl': PickleLoader(), 'pickle': PickleLoader(), ...}

    ```
    """
    from iden import io  # Local import to avoid cyclic dependencies

    return (
        io.json.get_loader_mapping()
        | io.pickle.get_loader_mapping()
        | io.text.get_loader_mapping()
        | io.torch.get_loader_mapping()
        | io.yaml.get_loader_mapping()
    )
