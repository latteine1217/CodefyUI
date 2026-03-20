from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class DropoutNode(BaseNode):
    NODE_NAME = "Dropout"
    CATEGORY = "CNN"
    DESCRIPTION = "Apply dropout regularization to input tensor (wraps nn.Dropout)"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Tensor with dropout applied"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="p",
                param_type=ParamType.FLOAT,
                default=0.5,
                description="Probability of an element to be zeroed",
                min_value=0.0,
                max_value=1.0,
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        p = params.get("p", 0.5)

        dropout = nn.Dropout(p=p)
        output = dropout(tensor)
        return {"tensor": output}
