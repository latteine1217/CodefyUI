from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class LayerNormNode(BaseNode):
    NODE_NAME = "LayerNorm"
    CATEGORY = "Normalization"
    DESCRIPTION = "Apply layer normalization (wraps nn.LayerNorm). Essential for Transformer architectures."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Normalized output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="normalized_shape",
                param_type=ParamType.STRING,
                default="512",
                description="Shape to normalize over as comma-separated ints (e.g. '512' or '64,32')",
            ),
            ParamDefinition(name="eps", param_type=ParamType.FLOAT, default=1e-5, description="Epsilon for numerical stability"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        shape_str = params.get("normalized_shape", "512")
        normalized_shape = [int(s.strip()) for s in shape_str.split(",")]
        eps = params.get("eps", 1e-5)

        ln = nn.LayerNorm(normalized_shape=normalized_shape, eps=eps)
        return {"tensor": ln(tensor)}
