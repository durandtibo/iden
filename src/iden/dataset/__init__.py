r"""Contain dataset implementations."""

from __future__ import annotations

__all__ = ["BaseDataset", "VanillaDataset", "create_vanilla_dataset"]

from iden.dataset.base import BaseDataset
from iden.dataset.vanilla import VanillaDataset, create_vanilla_dataset
