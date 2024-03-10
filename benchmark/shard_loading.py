from __future__ import annotations

import logging
from pathlib import Path

import torch

from iden.dataset import BaseDataset, create_vanilla_dataset, load_from_uri
from iden.shard import (
    BaseShard,
    ShardDict,
    ShardTuple,
    create_shard_dict,
    create_shard_tuple,
    create_torch_safetensors_shard,
)
from iden.utils.path import sanitize_path
from iden.utils.time import timeblock

logger = logging.getLogger(__name__)


def create_shard(uri: str, path_data: Path) -> BaseShard:
    batch_size = 1000000
    return create_torch_safetensors_shard(
        data={
            "key1": torch.empty(batch_size, 64),
            "key2": torch.empty(batch_size),
            "key3": torch.empty(batch_size, 128, 2),
        },
        uri=uri,
        path=path_data,
    )


def create_shard_split(uri: str, path_data: Path) -> ShardTuple:
    num_shards = 5
    shards = []
    for i in range(1, num_shards + 1):
        sid = f"{i:04}"
        uri_shard = sanitize_path(uri).parent.joinpath(f"uri_shard_{sid}").as_uri()
        shards.append(create_shard(uri=uri_shard, path_data=path_data.joinpath(sid)))
    return create_shard_tuple(uri=uri, shards=shards)


def create_shards(uri: str, path_data: Path) -> ShardDict:
    splits = ["train"]
    shards = {}
    for split in splits:
        uri_split = sanitize_path(uri).parent.joinpath(f"{split}/uri_split").as_uri()
        shards[split] = create_shard_split(uri_split, path_data=path_data.joinpath(split))
    return create_shard_dict(uri=uri, shards=shards)


def create_assets(uri: str, path_data: Path) -> ShardDict:
    return create_shard_dict(uri=uri, shards={})


def create_dataset(uri: str, path_data: Path) -> BaseDataset:
    shards = create_shards(
        uri=sanitize_path(uri).parent.joinpath("shards/uri_shards").as_uri(),
        path_data=path_data.joinpath("shards"),
    )
    assets = create_assets(
        uri=sanitize_path(uri).parent.joinpath("assets/uri_assets").as_uri(),
        path_data=path_data.joinpath("assets"),
    )
    return create_vanilla_dataset(shards=shards, assets=assets, uri=uri)


def get_dataset(uri: str, path_data: Path) -> BaseDataset:
    uri_file = sanitize_path(uri)
    if not uri_file.is_file():
        create_dataset(uri=uri, path_data=path_data)
    logger.info(f"loading dataset from {uri}")
    return load_from_uri(uri)


def benchmark_shard_loading(dataset: BaseDataset) -> None:
    with timeblock():
        total = 0
        shards = dataset.get_shards("train")
        for shard in shards:
            data = shard.get_data()
            total += data["key1"].shape[0]
        logger.info(f"total: {total:,}")


def main() -> None:
    path = Path.cwd().joinpath("tmp/dataset/safetensors1")
    uri = path.joinpath("uri/dataset").as_uri()
    path_data = path.joinpath("data")
    dataset = get_dataset(uri=uri, path_data=path_data)
    logger.info(f"dataset:\n{dataset}")

    benchmark_shard_loading(dataset)
    benchmark_shard_loading(dataset)
    benchmark_shard_loading(dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
