from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class LRSchedulerNode(BaseNode):
    NODE_NAME = "LRScheduler"
    CATEGORY = "Training"
    DESCRIPTION = "Create a learning rate scheduler for an optimizer"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer to schedule"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="scheduler", data_type=DataType.ANY, description="Configured LR scheduler"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="type",
                param_type=ParamType.SELECT,
                default="StepLR",
                description="Scheduler type",
                options=["StepLR", "CosineAnnealingLR", "ExponentialLR", "ReduceLROnPlateau", "CosineAnnealingWarmRestarts", "MultiStepLR", "OneCycleLR"],
            ),
            ParamDefinition(name="step_size", param_type=ParamType.INT, default=10, description="Step size for StepLR"),
            ParamDefinition(name="gamma", param_type=ParamType.FLOAT, default=0.1, description="Decay factor (for StepLR/ExponentialLR)"),
            ParamDefinition(name="T_max", param_type=ParamType.INT, default=50, description="Max iterations for CosineAnnealingLR"),
            ParamDefinition(name="max_lr", param_type=ParamType.FLOAT, default=0.01, description="Max learning rate for OneCycleLR"),
            ParamDefinition(name="total_steps", param_type=ParamType.INT, default=1000, description="Total training steps for OneCycleLR"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.optim.lr_scheduler as lr_scheduler

        optimizer = inputs["optimizer"]
        sched_type = params.get("type", "StepLR")

        if sched_type == "StepLR":
            sched = lr_scheduler.StepLR(optimizer, step_size=params.get("step_size", 10), gamma=params.get("gamma", 0.1))
        elif sched_type == "CosineAnnealingLR":
            sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.get("T_max", 50))
        elif sched_type == "ExponentialLR":
            sched = lr_scheduler.ExponentialLR(optimizer, gamma=params.get("gamma", 0.1))
        elif sched_type == "ReduceLROnPlateau":
            sched = lr_scheduler.ReduceLROnPlateau(optimizer, factor=params.get("gamma", 0.1))
        elif sched_type == "CosineAnnealingWarmRestarts":
            sched = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=params.get("T_max", 50))
        elif sched_type == "MultiStepLR":
            milestones = [params.get("step_size", 10) * i for i in range(1, 5)]
            sched = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=params.get("gamma", 0.1))
        elif sched_type == "OneCycleLR":
            sched = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=params.get("max_lr", 0.01),
                total_steps=params.get("total_steps", 1000),
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")

        return {"scheduler": sched}
