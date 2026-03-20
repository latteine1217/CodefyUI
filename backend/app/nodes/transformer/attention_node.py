from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class MultiHeadAttentionNode(BaseNode):
    NODE_NAME = "MultiHeadAttention"
    CATEGORY = "Transformer"
    DESCRIPTION = "Apply multi-head attention mechanism (wraps nn.MultiheadAttention)"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="query", data_type=DataType.TENSOR, description="Query tensor (seq_len, batch, embed_dim)"),
            PortDefinition(name="key", data_type=DataType.TENSOR, description="Key tensor (seq_len, batch, embed_dim)"),
            PortDefinition(name="value", data_type=DataType.TENSOR, description="Value tensor (seq_len, batch, embed_dim)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="output", data_type=DataType.TENSOR, description="Attention output tensor"),
            PortDefinition(name="weights", data_type=DataType.TENSOR, description="Attention weight tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="embed_dim", param_type=ParamType.INT, default=512, description="Total dimension of the model"),
            ParamDefinition(name="num_heads", param_type=ParamType.INT, default=8, description="Number of parallel attention heads"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        query = inputs["query"]
        key = inputs["key"]
        value = inputs["value"]
        embed_dim = params.get("embed_dim", 512)
        num_heads = params.get("num_heads", 8)

        mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        output, weights = mha(query, key, value)
        return {"output": output, "weights": weights}
