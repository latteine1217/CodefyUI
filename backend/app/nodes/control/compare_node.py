from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class CompareNode(BaseNode):
    NODE_NAME = "Compare"
    CATEGORY = "Control"
    DESCRIPTION = "Compare two scalar values. Outputs 1.0 (true) or 0.0 (false)."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="a", data_type=DataType.SCALAR, description="First value"),
            PortDefinition(name="b", data_type=DataType.SCALAR, description="Second value"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="result", data_type=DataType.SCALAR, description="1.0 if true, 0.0 if false"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="operation",
                param_type=ParamType.SELECT,
                default="==",
                description="Comparison operator",
                options=["==", "!=", "<", ">", "<=", ">="],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        a = inputs.get("a", 0)
        b = inputs.get("b", 0)
        # Handle tensor scalars
        if hasattr(a, "item"):
            a = a.item()
        if hasattr(b, "item"):
            b = b.item()
        a, b = float(a), float(b)

        op = params.get("operation", "==")
        ops = {
            "==": a == b,
            "!=": a != b,
            "<": a < b,
            ">": a > b,
            "<=": a <= b,
            ">=": a >= b,
        }
        return {"result": 1.0 if ops.get(op, False) else 0.0}
