from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class OptimizerNode(BaseNode):
    NODE_NAME = "Optimizer"
    CATEGORY = "Training"
    DESCRIPTION = "Create an optimizer (Adam, SGD, or AdamW) for model parameters"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model whose parameters to optimize"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Configured optimizer"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="type",
                param_type=ParamType.SELECT,
                default="Adam",
                description="Optimizer algorithm",
                options=["Adam", "SGD", "AdamW"],
            ),
            ParamDefinition(name="lr", param_type=ParamType.FLOAT, default=0.001, description="Learning rate", min_value=0.0),
            ParamDefinition(name="weight_decay", param_type=ParamType.FLOAT, default=0.0, description="Weight decay (L2 penalty)", min_value=0.0),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.optim as optim

        model = inputs["model"]
        opt_type = params.get("type", "Adam")
        lr = params.get("lr", 0.001)
        weight_decay = params.get("weight_decay", 0.0)

        optimizer_map = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            "AdamW": optim.AdamW,
        }

        optimizer_cls = optimizer_map.get(opt_type)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

        optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

        return {"optimizer": optimizer}
