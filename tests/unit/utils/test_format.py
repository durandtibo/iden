from __future__ import annotations

import pytest

from iden.utils.format import human_time

################################
#     Tests for human_time     #
################################


@pytest.mark.parametrize(
    ("seconds", "human"),
    [
        (1, "0:00:01"),
        (61, "0:01:01"),
        (3661, "1:01:01"),
        (3661.0, "1:01:01"),
        (1.1, "0:00:01.100000"),
        (3600 * 24 + 3661, "1 day, 1:01:01"),
        (3600 * 48 + 3661, "2 days, 1:01:01"),
    ],
)
def test_human_time(seconds: float, human: str) -> None:
    assert human_time(seconds) == human
