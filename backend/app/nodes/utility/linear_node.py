from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class LinearNode(BaseNode):
    NODE_NAME = "Linear"
    CATEGORY = "Utility"
    DESCRIPTION = "Fully-connected (dense) layer: nn.Linear(in_features, out_features)"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (..., in_features)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Output tensor (..., out_features)"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="in_features", param_type=ParamType.INT, default=512, description="Input feature size"),
            ParamDefinition(name="out_features", param_type=ParamType.INT, default=10, description="Output feature size"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        linear = nn.Linear(
            in_features=params.get("in_features", 512),
            out_features=params.get("out_features", 10),
        )
        return {"tensor": linear(tensor)}
