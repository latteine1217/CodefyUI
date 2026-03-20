from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class LSTMNode(BaseNode):
    NODE_NAME = "LSTM"
    CATEGORY = "RNN"
    DESCRIPTION = "Apply LSTM recurrent layer to input sequence (wraps nn.LSTM)"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (batch, seq_len, input_size) if batch_first=True"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="output", data_type=DataType.TENSOR, description="Output tensor containing hidden states for each time step"),
            PortDefinition(name="hidden", data_type=DataType.TENSOR, description="Final hidden state (h_n)"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="input_size", param_type=ParamType.INT, default=128, description="Number of expected features in the input"),
            ParamDefinition(name="hidden_size", param_type=ParamType.INT, default=256, description="Number of features in the hidden state"),
            ParamDefinition(name="num_layers", param_type=ParamType.INT, default=1, description="Number of recurrent layers"),
            ParamDefinition(name="batch_first", param_type=ParamType.BOOL, default=True, description="If True, input/output shape is (batch, seq, feature)"),
            ParamDefinition(name="bidirectional", param_type=ParamType.BOOL, default=False, description="If True, becomes a bidirectional LSTM"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        input_size = params.get("input_size", 128)
        hidden_size = params.get("hidden_size", 256)
        num_layers = params.get("num_layers", 1)
        batch_first = params.get("batch_first", True)
        bidirectional = params.get("bidirectional", False)

        lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        output, (h_n, _c_n) = lstm(tensor)
        return {"output": output, "hidden": h_n}
