from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class PrintNode(BaseNode):
    NODE_NAME = "Print"
    CATEGORY = "Utility"
    DESCRIPTION = "Print input value to console and pass through"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY, description="Any value to print")]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY, description="Pass-through")]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="label", param_type=ParamType.STRING, default="", description="Label prefix"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        value = inputs.get("value")
        label = params.get("label", "")
        prefix = f"[{label}] " if label else ""
        text = f"{prefix}{value}"
        print(text)
        return {"value": value, "__log__": text}
