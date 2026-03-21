from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class EmbeddingNode(BaseNode):
    NODE_NAME = "Embedding"
    CATEGORY = "Utility"
    DESCRIPTION = "Lookup embedding vectors for integer indices (wraps nn.Embedding). Used in NLP and sequence models."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor of integer indices (LongTensor)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Embedding output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="num_embeddings", param_type=ParamType.INT, default=10000, description="Size of the vocabulary"),
            ParamDefinition(name="embedding_dim", param_type=ParamType.INT, default=256, description="Dimension of each embedding vector"),
            ParamDefinition(name="padding_idx", param_type=ParamType.INT, default=-1, description="Index for padding token (-1 for none)"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        num_embeddings = params.get("num_embeddings", 10000)
        embedding_dim = params.get("embedding_dim", 256)
        padding_idx = params.get("padding_idx", -1)

        kwargs = {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
        }
        if padding_idx >= 0:
            kwargs["padding_idx"] = padding_idx

        emb = nn.Embedding(**kwargs)
        return {"tensor": emb(tensor)}
