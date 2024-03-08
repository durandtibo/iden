r"""Contain shard implementations."""

from __future__ import annotations

__all__ = [
    "BaseShard",
    "FileShard",
    "InMemoryShard",
    "JsonShard",
    "PickleShard",
    "TorchSafetensorsShard",
    "create_json_shard",
    "create_pickle_shard",
    "create_torch_safetensors_shard",
    "load_from_uri",
]

from iden.shard.base import BaseShard
from iden.shard.file import FileShard
from iden.shard.in_memory import InMemoryShard
from iden.shard.json import JsonShard, create_json_shard
from iden.shard.loading import load_from_uri
from iden.shard.pickle import PickleShard, create_pickle_shard
from iden.shard.safetensors import TorchSafetensorsShard, create_torch_safetensors_shard
