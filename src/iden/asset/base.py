r"""Contain the base class to implement an asset object."""

from __future__ import annotations

__all__ = ["BaseAsset"]

from abc import ABC, abstractmethod
from typing import Any


class BaseAsset(ABC):
    r"""Define the base class to implement an asset."""

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two assets are equal or not.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

        Returns:
            ``True`` if the two assets are equal, otherwise ``False``.
        """

    @abstractmethod
    def get_asset(self, asset_id: str) -> Any:
        r"""Get a data asset.

        Args:
            asset_id: The asset ID used to find the asset.

        Returns:
            The asset.

        Raises:
            AssetNotFoundError: if the asset does not exist.
        """

    @abstractmethod
    def has_asset(self, asset_id: str) -> bool:
        r"""Indicate if the asset exists or not.

        Args:
            asset_id: The asset ID used to find the asset.

        Returns:
            ``True`` if the asset exists, otherwise ``False``.
        """

    @abstractmethod
    def get_uri(self) -> str:
        r"""Get the Uniform Resource Identifier (URI) of the asset.

        Returns:
            The asset's URI.
        """
