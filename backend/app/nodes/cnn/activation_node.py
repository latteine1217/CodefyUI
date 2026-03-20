from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class ActivationNode(BaseNode):
    NODE_NAME = "Activation"
    CATEGORY = "CNN"
    DESCRIPTION = "Apply activation function (ReLU, Sigmoid, or Tanh) to input tensor"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Activated output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="function",
                param_type=ParamType.SELECT,
                default="relu",
                description="Activation function to apply",
                options=["relu", "sigmoid", "tanh"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        tensor = inputs["tensor"]
        function = params.get("function", "relu")

        if function == "relu":
            output = F.relu(tensor)
        elif function == "sigmoid":
            output = torch.sigmoid(tensor)
        elif function == "tanh":
            output = torch.tanh(tensor)
        else:
            raise ValueError(f"Unsupported activation function: {function}")

        return {"tensor": output}
