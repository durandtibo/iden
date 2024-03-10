r"""Contain code to load a dataset from its Uniform Resource Identifier
(URI)."""

from __future__ import annotations

__all__ = ["load_from_uri"]

from typing import TYPE_CHECKING

from iden.constants import LOADER
from iden.dataset.loader.base import setup_dataset_loader
from iden.io import load_json
from iden.utils.path import sanitize_path

if TYPE_CHECKING:
    from iden.dataset import BaseDataset


def load_from_uri(uri: str) -> BaseDataset:
    r"""Load a dataset from its Uniform Resource Identifier (URI).

    Args:
        uri: The URI of the dataset.

    Returns:
        The dataset associated to the URI.

    Raises:
        FileNotFoundError: if the URI file does not exist.
    """
    path = sanitize_path(uri)
    if not path.is_file():
        msg = f"uri file does not exist: {path}"
        raise FileNotFoundError(msg)
    config = load_json(path)
    loader = setup_dataset_loader(config[LOADER])
    return loader.load(uri)
