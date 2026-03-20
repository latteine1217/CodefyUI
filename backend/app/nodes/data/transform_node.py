from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class TransformNode(BaseNode):
    NODE_NAME = "Transform"
    CATEGORY = "Data"
    DESCRIPTION = "Apply common image transforms (resize, normalize, to_tensor) to a dataset"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="dataset", data_type=DataType.DATASET, description="Dataset to apply transforms to"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="dataset", data_type=DataType.DATASET, description="Dataset with updated transforms"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="resize", param_type=ParamType.INT, default=0, description="Resize dimension (0 means no resize)", min_value=0),
            ParamDefinition(name="normalize", param_type=ParamType.BOOL, default=True, description="Apply normalization (mean=0.5, std=0.5)"),
            ParamDefinition(name="to_tensor", param_type=ParamType.BOOL, default=True, description="Convert PIL images to tensors"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        from torchvision import transforms

        dataset = inputs["dataset"]
        resize = params.get("resize", 0)
        normalize = params.get("normalize", True)
        to_tensor = params.get("to_tensor", True)

        transform_list = []

        if resize > 0:
            transform_list.append(transforms.Resize((resize, resize)))

        if to_tensor:
            transform_list.append(transforms.ToTensor())

        if normalize:
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))

        if transform_list:
            dataset.transform = transforms.Compose(transform_list)

        return {"dataset": dataset}
