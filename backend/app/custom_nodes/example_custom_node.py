"""
Example custom node for CodefyUI.

Drop any .py file in this directory that extends BaseNode,
and it will be auto-discovered on startup or when you call
POST /api/nodes/reload.
"""

from typing import Any

from ..core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class AddScalarNode(BaseNode):
    NODE_NAME = "AddScalar"
    CATEGORY = "Custom"
    DESCRIPTION = "Add a scalar value to a tensor (example custom node)"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="tensor + scalar"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="value", param_type=ParamType.FLOAT, default=1.0, description="Scalar to add"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        tensor = inputs["tensor"]
        value = params.get("value", 1.0)
        return {"tensor": tensor + value}
