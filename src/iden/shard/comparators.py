r"""Contain some comparators to use ``BaseShard`` objects with
``coola.objects_are_equal``."""

from __future__ import annotations

__all__ = ["ShardEqualityComparator"]

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from coola.equality.comparators import BaseEqualityComparator
from coola.equality.handlers import EqualNanHandler, SameObjectHandler, SameTypeHandler
from coola.equality.testers import EqualityTester

from iden.shard.base import BaseShard

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)

S = TypeVar("S", bound="ShardEqualityComparator")


class ShardEqualityComparator(BaseEqualityComparator[BaseShard[Any]]):  # noqa: PLW1641
    r"""Implement an equality comparator for ``BaseShard`` objects."""

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(EqualNanHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> S:
        return self.__class__()

    def equal(self, actual: BaseShard[Any], expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


if not EqualityTester.has_comparator(BaseShard):  # pragma: no cover
    EqualityTester.add_comparator(BaseShard, ShardEqualityComparator())
