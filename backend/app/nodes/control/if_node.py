from typing import Any

from ...core.node_base import BaseNode, DataType, PortDefinition


class IfNode(BaseNode):
    NODE_NAME = "If"
    CATEGORY = "Control"
    DESCRIPTION = "Select between two inputs based on a condition. Nonzero = true, zero = false. Note: both branches are evaluated before selection."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="condition", data_type=DataType.SCALAR, description="Condition value (nonzero = true)"),
            PortDefinition(name="if_true", data_type=DataType.ANY, description="Output when condition is true"),
            PortDefinition(name="if_false", data_type=DataType.ANY, description="Output when condition is false"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="output", data_type=DataType.ANY, description="Selected value"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        condition = inputs.get("condition", 0)
        if hasattr(condition, "item"):
            condition = condition.item()

        if condition:
            return {"output": inputs.get("if_true")}
        else:
            return {"output": inputs.get("if_false")}
