"""HuggingFaceDataset node — load image-classification datasets from HF Hub."""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class HuggingFaceDatasetNode(BaseNode):
    NODE_NAME = "HuggingFaceDataset"
    CATEGORY = "Data"
    DESCRIPTION = "Load an image-classification dataset from HuggingFace Hub"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Dataset wrapped as a torch.utils.data.Dataset",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="dataset_name",
                param_type=ParamType.STRING,
                default="ylecun/mnist",
                description="HuggingFace Hub repo id (e.g. cifar10, uoft-cs/cifar100)",
            ),
            ParamDefinition(
                name="subset",
                param_type=ParamType.STRING,
                default="",
                description="Config name for multi-config datasets (empty = no subset)",
            ),
            ParamDefinition(
                name="split",
                param_type=ParamType.STRING,
                default="train",
                description="Split: train/test/validation, or HF slice syntax (train[:1000])",
            ),
            ParamDefinition(
                name="image_column",
                param_type=ParamType.STRING,
                default="image",
                description="Column name for the image feature",
            ),
            ParamDefinition(
                name="label_column",
                param_type=ParamType.STRING,
                default="label",
                description="Column name for the integer label",
            ),
            ParamDefinition(
                name="cache_dir",
                param_type=ParamType.STRING,
                default="",
                description="Override HF cache directory (empty = HF default)",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        from datasets import load_dataset
        from torchvision import transforms as T

        from ._hf_adapter import HFTorchImageDataset

        dataset_name = params.get("dataset_name", "ylecun/mnist")
        subset = params.get("subset", "") or None
        split = params.get("split", "train")
        image_column = params.get("image_column", "image")
        label_column = params.get("label_column", "label")
        cache_dir = params.get("cache_dir", "") or None

        ds = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)

        wrapped = HFTorchImageDataset(
            ds,
            image_column=image_column,
            label_column=label_column,
            transform=T.ToTensor(),
        )
        return {"dataset": wrapped}
