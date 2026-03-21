from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class ConvTranspose2dNode(BaseNode):
    NODE_NAME = "ConvTranspose2d"
    CATEGORY = "CNN"
    DESCRIPTION = "Apply 2D transposed convolution (deconvolution) to input tensor (wraps nn.ConvTranspose2d)"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C, H, W)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Upsampled output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="in_channels", param_type=ParamType.INT, default=64, description="Number of input channels"),
            ParamDefinition(name="out_channels", param_type=ParamType.INT, default=32, description="Number of output channels"),
            ParamDefinition(name="kernel_size", param_type=ParamType.INT, default=2, description="Size of the convolving kernel"),
            ParamDefinition(name="stride", param_type=ParamType.INT, default=2, description="Stride of the convolution"),
            ParamDefinition(name="padding", param_type=ParamType.INT, default=0, description="Zero-padding added to both sides"),
            ParamDefinition(name="output_padding", param_type=ParamType.INT, default=0, description="Additional size added to output shape"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        conv_t = nn.ConvTranspose2d(
            in_channels=params.get("in_channels", 64),
            out_channels=params.get("out_channels", 32),
            kernel_size=params.get("kernel_size", 2),
            stride=params.get("stride", 2),
            padding=params.get("padding", 0),
            output_padding=params.get("output_padding", 0),
        )
        return {"tensor": conv_t(tensor)}
