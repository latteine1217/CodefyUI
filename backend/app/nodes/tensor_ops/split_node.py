from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class SplitNode(BaseNode):
    NODE_NAME = "Split"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Split a tensor into chunks along a dimension"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor to split"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="chunk_0", data_type=DataType.TENSOR, description="First chunk"),
            PortDefinition(name="chunk_1", data_type=DataType.TENSOR, description="Second chunk"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="chunks", param_type=ParamType.INT, default=2, description="Number of chunks to split into"),
            ParamDefinition(name="dim", param_type=ParamType.INT, default=0, description="Dimension along which to split"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        tensor = inputs["tensor"]
        chunks = params.get("chunks", 2)
        dim = params.get("dim", 0)
        parts = torch.chunk(tensor, chunks, dim=dim)
        result = {}
        for i, part in enumerate(parts):
            result[f"chunk_{i}"] = part
        return result
