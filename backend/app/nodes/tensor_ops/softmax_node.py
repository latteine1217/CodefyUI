from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class SoftmaxNode(BaseNode):
    NODE_NAME = "Softmax"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Apply softmax function along a dimension"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (logits)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Softmax probabilities"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="dim", param_type=ParamType.INT, default=-1, description="Dimension along which to apply softmax"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn.functional as F

        tensor = inputs["tensor"]
        dim = params.get("dim", -1)
        return {"tensor": F.softmax(tensor, dim=dim)}
