import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class InferenceNode(BaseNode):
    NODE_NAME = "Inference"
    CATEGORY = "IO"
    DESCRIPTION = "Run inference (forward pass) on a trained model. Sets model to eval mode and disables gradients."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Trained model"),
            PortDefinition(name="input", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="output", data_type=DataType.TENSOR, description="Model prediction"),
            PortDefinition(name="model", data_type=DataType.MODEL, description="Pass-through model"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="cpu",
                description="Device to run inference on",
                options=["cpu", "cuda", "mps"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        model = inputs["model"]
        input_tensor = inputs["input"]
        device = params.get("device", "cpu")

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            device = "cpu"

        model = model.to(device)
        input_tensor = input_tensor.to(device)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        logger.info("Inference complete — input %s → output %s", list(input_tensor.shape), list(output.shape))

        return {"output": output, "model": model}
