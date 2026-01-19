from __future__ import annotations

from coola.equality.tester import get_default_registry

from iden.dataset import BaseDataset


def test_equality_tester_registry_has_equality_tester() -> None:
    assert get_default_registry().has_equality_tester(BaseDataset)
