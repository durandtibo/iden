r"""Contain in-memory asset implementations."""

from __future__ import annotations

__all__ = ["InMemoryAsset"]

from typing import Any

from coola import objects_are_equal

from iden.asset.base import BaseAsset
from iden.dataset.exceptions import AssetNotFoundError


class InMemoryAsset(BaseAsset):
    r"""Implement an in-memory asset.

    Args:
        uri: The asset URI.
        data: The asset data.

    Example usage:

    ```pycon
    >>> from iden.asset import InMemoryAsset
    >>> asset = InMemoryAsset(uri="my_uri", data={"mean": 42})
    >>> asset
    InMemoryAsset(uri=my_uri, num_assets=1)
    >>> asset.get_asset("mean")
    42

    ```
    """

    def __init__(self, uri: str, data: dict) -> None:
        self._uri = uri
        self._data = data

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(uri={self._uri}, num_assets={len(self._data):,})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(
            self.get_uri(), other.get_uri(), equal_nan=equal_nan
        ) and objects_are_equal(self._data, other._data, equal_nan=equal_nan)

    def get_asset(self, asset_id: str) -> Any:
        if asset_id not in self._data:
            msg = f"asset '{asset_id}' does not exist"
            raise AssetNotFoundError(msg)
        return self._data[asset_id]

    def has_asset(self, asset_id: str) -> bool:
        return asset_id in self._data

    def get_uri(self) -> str:
        return self._uri
