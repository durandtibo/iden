from __future__ import annotations

from coola.equality.testers import EqualityTester

from iden.dataset import BaseDataset


def test_equality_tester_has_comparator() -> None:
    assert EqualityTester.has_comparator(BaseDataset)
