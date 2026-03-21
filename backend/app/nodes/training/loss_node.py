from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class LossNode(BaseNode):
    NODE_NAME = "Loss"
    CATEGORY = "Training"
    DESCRIPTION = "Create a loss function"

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
                options=["CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "L1Loss", "SmoothL1Loss", "NLLLoss", "KLDivLoss", "HuberLoss", "BCELoss", "MarginRankingLoss", "CosineEmbeddingLoss"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        loss_type = params.get("type", "CrossEntropyLoss")

        loss_map = {
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "MSELoss": nn.MSELoss,
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
            "L1Loss": nn.L1Loss,
            "SmoothL1Loss": nn.SmoothL1Loss,
            "NLLLoss": nn.NLLLoss,
            "KLDivLoss": nn.KLDivLoss,
            "HuberLoss": nn.HuberLoss,
            "BCELoss": nn.BCELoss,
            "MarginRankingLoss": nn.MarginRankingLoss,
            "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
        }

        loss_cls = loss_map.get(loss_type)
        if loss_cls is None:
            raise ValueError(f"Unsupported loss function type: {loss_type}")

        loss_fn = loss_cls()

        return {"loss_fn": loss_fn}
