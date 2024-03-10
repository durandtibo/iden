from __future__ import annotations

from iden.dataset import BaseDataset, create_vanilla_dataset
from iden.utils.path import sanitize_path


def create_dataset(uri: str) -> BaseDataset:
    shards = ...
    assets = ...

    return create_vanilla_dataset(shards=shards, assets=assets, uri=uri)


def get_dataset(uri: str) -> BaseDataset:
    uri_file = sanitize_path(uri)
    if not uri_file.is_file():
        create_dataset(uri)
    return load_from_uri(uri)
