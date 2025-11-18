# noqa: INP001
r"""Script to create or update the package versions."""

from __future__ import annotations

import logging
from pathlib import Path

from feu.utils.io import save_json
from feu.version import (
    get_latest_minor_versions,
    get_versions,
)

logger = logging.getLogger(__name__)


def get_package_versions() -> dict[str, list[str]]:
    r"""Get the versions for each package.

    Returns:
        A dictionary with the versions for each package.
    """
    return {
        "coola": list(get_versions("coola", lower="0.9.1")),
        "objectory": list(get_versions("objectory", lower="0.2.1")),
        "cloudpickle": list(get_latest_minor_versions("cloudpickle", lower="3.0")),
        "joblib": list(get_latest_minor_versions("joblib", lower="1.3")),
        "numpy": list(get_latest_minor_versions("numpy", lower="1.24")),
        "pyyaml": list(get_latest_minor_versions("pyyaml", lower="6.0")),
        "safetensors": list(get_latest_minor_versions("safetensors", lower="0.4")),
        "torch": list(get_latest_minor_versions("torch", lower="2.0")),
    }


def main() -> None:
    r"""Generate the package versions and save them in a JSON file."""
    versions = get_package_versions()
    logger.info(f"{versions=}")
    path = Path(__file__).parent.parent.joinpath("dev/config").joinpath("package_versions.json")
    logger.info(f"Saving package versions to {path}")
    save_json(versions, path, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
