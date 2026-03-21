from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class GroupNormNode(BaseNode):
    NODE_NAME = "GroupNorm"
    CATEGORY = "Normalization"
    DESCRIPTION = "Apply group normalization (wraps nn.GroupNorm). Used in modern CNN architectures."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C, *)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Normalized output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="num_groups", param_type=ParamType.INT, default=32, description="Number of groups to divide channels into"),
            ParamDefinition(name="num_channels", param_type=ParamType.INT, default=256, description="Number of channels (must be divisible by num_groups)"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        gn = nn.GroupNorm(
            num_groups=params.get("num_groups", 32),
            num_channels=params.get("num_channels", 256),
        )
        return {"tensor": gn(tensor)}
