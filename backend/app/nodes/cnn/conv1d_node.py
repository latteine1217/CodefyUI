from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class Conv1dNode(BaseNode):
    NODE_NAME = "Conv1d"
    CATEGORY = "CNN"
    DESCRIPTION = "Apply 1D convolution to input tensor (wraps nn.Conv1d)"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C, L)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Convolved output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="in_channels", param_type=ParamType.INT, default=1, description="Number of input channels"),
            ParamDefinition(name="out_channels", param_type=ParamType.INT, default=32, description="Number of output channels"),
            ParamDefinition(name="kernel_size", param_type=ParamType.INT, default=3, description="Size of the convolving kernel"),
            ParamDefinition(name="stride", param_type=ParamType.INT, default=1, description="Stride of the convolution"),
            ParamDefinition(name="padding", param_type=ParamType.INT, default=1, description="Zero-padding added to both sides"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        conv = nn.Conv1d(
            in_channels=params.get("in_channels", 1),
            out_channels=params.get("out_channels", 32),
            kernel_size=params.get("kernel_size", 3),
            stride=params.get("stride", 1),
            padding=params.get("padding", 1),
        )
        return {"tensor": conv(tensor)}
