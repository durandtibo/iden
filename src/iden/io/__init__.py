r"""Contain data loaders and savers."""

from __future__ import annotations

__all__ = [
    "BaseFileSaver",
    "BaseLoader",
    "BaseSaver",
    "JsonLoader",
    "JsonSaver",
    "PickleLoader",
    "PickleSaver",
    "TextLoader",
    "TextSaver",
    "load_json",
    "load_pickle",
    "load_text",
    "save_json",
    "save_pickle",
    "save_text",
]

from iden.io.base import BaseFileSaver, BaseLoader, BaseSaver
from iden.io.json import JsonLoader, JsonSaver, load_json, save_json
from iden.io.pickle import PickleLoader, PickleSaver, load_pickle, save_pickle
from iden.io.text import TextLoader, TextSaver, load_text, save_text
