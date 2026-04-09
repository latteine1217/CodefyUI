"""KaggleDataset node — download and load a Kaggle dataset as an ImageFolder."""

from __future__ import annotations

import os
from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class KaggleDatasetNode(BaseNode):
    NODE_NAME = "KaggleDataset"
    CATEGORY = "Data"
    DESCRIPTION = "Download a Kaggle dataset and load it as an ImageFolder"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Dataset loaded as torchvision ImageFolder",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="dataset_slug",
                param_type=ParamType.STRING,
                default="",
                description="Kaggle dataset slug in 'owner/name' form",
            ),
            ParamDefinition(
                name="subdir",
                param_type=ParamType.STRING,
                default="",
                description="Relative path inside the downloaded dataset where ImageFolder structure begins",
            ),
            ParamDefinition(
                name="cache_dir",
                param_type=ParamType.STRING,
                default="",
                description="Override kagglehub cache directory (empty = default)",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        try:
            import kagglehub
        except ImportError as e:
            raise RuntimeError(
                "KaggleDataset requires the 'kagglehub' package. "
                "Install with: pip install kagglehub "
                "(or `uv sync --extra ml` from backend/)"
            ) from e

        from torchvision import transforms as T
        from torchvision.datasets import ImageFolder

        dataset_slug = params.get("dataset_slug", "")
        subdir = params.get("subdir", "")
        cache_dir = params.get("cache_dir", "")

        if cache_dir:
            os.environ["KAGGLEHUB_CACHE"] = cache_dir

        download_path = kagglehub.dataset_download(dataset_slug)
        root = os.path.join(download_path, subdir) if subdir else download_path

        return {"dataset": ImageFolder(root, transform=T.ToTensor())}
