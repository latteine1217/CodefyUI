from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class TransformerDecoderNode(BaseNode):
    NODE_NAME = "TransformerDecoder"
    CATEGORY = "Transformer"
    DESCRIPTION = "Apply Transformer decoder stack to input tensor with encoder memory"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Target tensor (seq_len, batch, d_model)"),
            PortDefinition(name="memory", data_type=DataType.TENSOR, description="Encoder output / memory tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Decoded output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="d_model", param_type=ParamType.INT, default=512, description="Dimension of the model"),
            ParamDefinition(name="nhead", param_type=ParamType.INT, default=8, description="Number of attention heads"),
            ParamDefinition(name="num_layers", param_type=ParamType.INT, default=6, description="Number of decoder layers"),
            ParamDefinition(name="dim_feedforward", param_type=ParamType.INT, default=2048, description="Dimension of feedforward network"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        memory = inputs["memory"]
        d_model = params.get("d_model", 512)
        nhead = params.get("nhead", 8)
        num_layers = params.get("num_layers", 6)
        dim_feedforward = params.get("dim_feedforward", 2048)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        output = decoder(tensor, memory)
        return {"tensor": output}
