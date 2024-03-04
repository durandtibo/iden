r"""Contain safetensors data loaders and savers."""

from __future__ import annotations

__all__ = ["NumpySafetensorsSaver", "TorchSafetensorsSaver"]

from iden.io.safetensors.savers import NumpySafetensorsSaver, TorchSafetensorsSaver
