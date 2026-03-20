from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class BatchNormNode(BaseNode):
    NODE_NAME = "BatchNorm2d"
    CATEGORY = "CNN"
    DESCRIPTION = "Apply 2D batch normalization to input tensor (wraps nn.BatchNorm2d)"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C, H, W)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Normalized output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="num_features", param_type=ParamType.INT, default=32, description="Number of features (channels) to normalize"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        num_features = params.get("num_features", 32)

        batchnorm = nn.BatchNorm2d(num_features=num_features)
        output = batchnorm(tensor)
        return {"tensor": output}
