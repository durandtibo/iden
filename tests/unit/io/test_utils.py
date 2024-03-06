from __future__ import annotations

from coola.testing import torch_available

from iden.io import JsonLoader, PickleLoader, TextLoader, TorchLoader, YamlLoader
from iden.io.utils import get_loader_mapping

########################################
#     Tests for get_loader_mapping     #
########################################


def test_get_loader_mapping() -> None:
    mapping = get_loader_mapping()
    assert len(mapping) >= 6
    assert isinstance(mapping["json"], JsonLoader)
    assert isinstance(mapping["pickle"], PickleLoader)
    assert isinstance(mapping["pkl"], PickleLoader)
    assert isinstance(mapping["txt"], TextLoader)
    assert isinstance(mapping["yaml"], YamlLoader)
    assert isinstance(mapping["yml"], YamlLoader)


@torch_available
def test_get_loader_mapping_torch() -> None:
    mapping = get_loader_mapping()
    assert isinstance(mapping["pt"], TorchLoader)
