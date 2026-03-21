import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class TrainingLoopNode(BaseNode):
    NODE_NAME = "TrainingLoop"
    CATEGORY = "Training"
    DESCRIPTION = "Run a training loop over a dataloader for a given number of epochs"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Model to train"),
            PortDefinition(name="dataloader", data_type=DataType.DATALOADER, description="Training data loader"),
            PortDefinition(name="optimizer", data_type=DataType.OPTIMIZER, description="Optimizer for parameter updates"),
            PortDefinition(name="loss_fn", data_type=DataType.LOSS_FN, description="Loss function"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="Trained model"),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="Loss history tensor (one value per epoch)"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="epochs", param_type=ParamType.INT, default=5, description="Number of training epochs", min_value=1),
            ParamDefinition(
                name="device",
                param_type=ParamType.SELECT,
                default="cpu",
                description="Device to train on",
                options=["cpu", "cuda", "mps"],
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        import torch

        model = inputs["model"]
        dataloader = inputs["dataloader"]
        optimizer = inputs["optimizer"]
        loss_fn = inputs["loss_fn"]
        epochs = params.get("epochs", 5)
        device = params.get("device", "cpu")

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            device = "cpu"

        model = model.to(device)
        loss_fn = loss_fn.to(device)

        # Re-bind optimizer to model's device-mapped parameters to ensure
        # consistency after model.to(device) (which may create new Parameter
        # objects on certain PyTorch versions / backends such as MPS).
        for param_group in optimizer.param_groups:
            param_group["params"] = list(model.parameters())

        model.train()

        # Collect training config for frontend display
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer_name = optimizer.__class__.__name__
        lr = optimizer.param_groups[0].get("lr", "N/A")
        loss_fn_name = loss_fn.__class__.__name__
        training_config = {
            "model_class": model.__class__.__name__,
            "params": param_count,
            "trainable": trainable_count,
            "optimizer": optimizer_name,
            "lr": lr,
            "loss_fn": loss_fn_name,
            "epochs": epochs,
            "device": device,
            "batch_size": getattr(dataloader, "batch_size", "N/A"),
        }

        if progress_callback:
            progress_callback({"event": "config", "config": training_config})

        epoch_losses = []

        for epoch in range(epochs):
            running_loss = 0.0
            batch_count = 0

            for batch_data in dataloader:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, targets = batch_data
                    data = data.to(device)
                    targets = targets.to(device)
                else:
                    data = batch_data.to(device) if hasattr(batch_data, "to") else batch_data
                    targets = None

                optimizer.zero_grad()

                outputs = model(data)

                if targets is not None:
                    loss = loss_fn(outputs, targets)
                else:
                    loss = loss_fn(outputs)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1

            avg_loss = running_loss / max(batch_count, 1)
            epoch_losses.append(avg_loss)
            logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, avg_loss)

            if progress_callback:
                progress_callback({
                    "event": "epoch",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "loss": round(avg_loss, 6),
                    "losses": [round(l, 6) for l in epoch_losses],
                })

        losses_tensor = torch.tensor(epoch_losses, dtype=torch.float32)

        return {"model": model, "losses": losses_tensor}
