from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class LossNode(BaseNode):
    NODE_NAME = "Loss"
    CATEGORY = "Training"
    DESCRIPTION = "Create a loss function (CrossEntropyLoss, MSELoss, or BCEWithLogitsLoss)"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="loss_fn", data_type=DataType.LOSS_FN, description="Loss function instance"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="type",
                param_type=ParamType.SELECT,
                default="CrossEntropyLoss",
                description="Loss function type",
                options=["CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        loss_type = params.get("type", "CrossEntropyLoss")

        loss_map = {
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "MSELoss": nn.MSELoss,
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        }

        loss_cls = loss_map.get(loss_type)
        if loss_cls is None:
            raise ValueError(f"Unsupported loss function type: {loss_type}")

        loss_fn = loss_cls()

        return {"loss_fn": loss_fn}
