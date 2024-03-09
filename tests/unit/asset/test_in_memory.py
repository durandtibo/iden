from __future__ import annotations

import pytest

from iden.asset import InMemoryAsset
from iden.dataset.exceptions import AssetNotFoundError


@pytest.fixture()
def asset() -> InMemoryAsset:
    return InMemoryAsset(uri="my_uri", data={"mean": 42})


###################################
#     Tests for InMemoryAsset     #
###################################


def test_in_memory_asset_str(asset: InMemoryAsset) -> None:
    assert str(asset).startswith("InMemoryAsset(")


def test_in_memory_asset_equal_true(asset: InMemoryAsset) -> None:
    assert asset.equal(InMemoryAsset(uri="my_uri", data={"mean": 42}))


def test_in_memory_asset_equal_false_different_uri() -> None:
    assert not InMemoryAsset(uri="uri1", data={}).equal(InMemoryAsset(uri="uri2", data={}))


def test_in_memory_asset_equal_false_different_asset(asset: InMemoryAsset) -> None:
    assert not asset.equal(InMemoryAsset(uri="my_uri", data={"mean": 2}))


def test_in_memory_asset_equal_false_different_type(asset: InMemoryAsset) -> None:
    assert not asset.equal("meow")


def test_in_memory_asset_equal_nan_true() -> None:
    assert InMemoryAsset(uri="my_uri", data={"mean": 42, "nan": float("nan")}).equal(
        InMemoryAsset(uri="my_uri", data={"mean": 42, "nan": float("nan")}), equal_nan=True
    )


def test_in_memory_asset_equal_nan_false() -> None:
    assert not InMemoryAsset(uri="my_uri", data={"mean": 42, "nan": float("nan")}).equal(
        InMemoryAsset(uri="my_uri", data={"mean": 42, "nan": float("nan")})
    )


def test_in_memory_asset_get_asset(asset: InMemoryAsset) -> None:
    assert asset.get_asset("mean") == 42


def test_in_memory_asset_get_asset_missing(asset: InMemoryAsset) -> None:
    with pytest.raises(AssetNotFoundError, match="asset 'missing' does not exist"):
        asset.get_asset("missing")


def test_in_memory_asset_has_asset_true(asset: InMemoryAsset) -> None:
    assert asset.has_asset("mean")


def test_in_memory_asset_has_asset_false(asset: InMemoryAsset) -> None:
    assert not asset.has_asset("missing")


def test_in_memory_asset_get_uri(asset: InMemoryAsset) -> None:
    assert asset.get_uri() == "my_uri"
