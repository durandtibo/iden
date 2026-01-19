from __future__ import annotations

import importlib
import logging
import tempfile
from pathlib import Path

from coola.equality import objects_are_equal

from iden.dataset import create_vanilla_dataset
from iden.io import JsonLoader, JsonSaver
from iden.shard import create_json_shard, create_shard_dict, create_shard_tuple

logger = logging.getLogger(__name__)


def check_imports() -> None:
    logger.info("Checking imports...")
    objects_to_import = [
        "iden.dataset.BaseDataset",
        "iden.dataset.loader.BaseDatasetLoader",
        "iden.io.BaseLoader",
        "iden.io.BaseSaver",
        "iden.shard.BaseShard",
        "iden.shard.loader.BaseShardLoader",
    ]
    for a in objects_to_import:
        module_path, name = a.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
        obj = getattr(module, name)
        assert obj is not None


def check_dataset() -> None:
    logger.info("Checking dataset...")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        shards = create_shard_dict(
            {
                "train": create_shard_tuple(
                    shards=[
                        create_json_shard(
                            data=[1, 2, 3],
                            uri=path.joinpath("train/uri1").as_uri(),
                            path=path.joinpath("train/data1.json"),
                        ),
                        create_json_shard(
                            data=[4, 5, 6],
                            uri=path.joinpath("train/uri2").as_uri(),
                            path=path.joinpath("train/data2.json"),
                        ),
                        create_json_shard(
                            data=[7, 8],
                            uri=path.joinpath("train/uri3").as_uri(),
                            path=path.joinpath("train/data3.json"),
                        ),
                    ],
                    uri=path.joinpath("uri_train").as_uri(),
                ),
                "val": create_shard_tuple(shards=[], uri=path.joinpath("uri_val").as_uri()),
                "test": create_shard_tuple(
                    shards=[
                        create_json_shard(
                            data=[10, 11, 12, 13, 14, 15],
                            uri=path.joinpath("test/uri1").as_uri(),
                            path=path.joinpath("test/data1.json"),
                        ),
                    ],
                    uri=path.joinpath("uri_test").as_uri(),
                ),
            },
            uri=path.joinpath("uri_shards").as_uri(),
        )
        assets = create_shard_dict(
            shards={
                "stats": create_json_shard(
                    data={"mean": 42},
                    uri=path.joinpath("uri_stats").as_uri(),
                    path=path.joinpath("data_stats.json"),
                )
            },
            uri=path.joinpath("uri_assets").as_uri(),
        )
        dataset = create_vanilla_dataset(
            shards=shards, assets=assets, uri=Path(tmpdir).joinpath("uri").as_uri()
        )
        assert dataset.get_splits() == {"train", "val", "test"}


def check_io_json() -> None:
    logger.info("Checking I/O - JSON...")
    with tempfile.TemporaryDirectory() as tmpdir:
        file = Path(tmpdir).joinpath("data.json")
        data = {"key1": [1, 2, 3], "key2": "abc"}
        JsonSaver().save(data, file)
        assert objects_are_equal(JsonLoader().load(file), data)


def check_shard_json() -> None:
    logger.info("Checking shard - JSON...")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {"key1": [1, 2, 3], "key2": "abc"}
        shard = create_json_shard(data, uri=Path(tmpdir).joinpath("uri").as_uri())
        assert objects_are_equal(shard.get_data(), data)


def main() -> None:
    check_imports()
    check_dataset()
    check_io_json()
    check_shard_json()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
