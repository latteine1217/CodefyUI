from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class DataLoaderNode(BaseNode):
    NODE_NAME = "DataLoader"
    CATEGORY = "Data"
    DESCRIPTION = "Wrap a dataset in a DataLoader for batched iteration"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="dataset", data_type=DataType.DATASET, description="Dataset to wrap"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="dataloader", data_type=DataType.DATALOADER, description="DataLoader for batched iteration"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="batch_size", param_type=ParamType.INT, default=32, description="Samples per batch", min_value=1),
            ParamDefinition(name="shuffle", param_type=ParamType.BOOL, default=True, description="Shuffle data at every epoch"),
            ParamDefinition(name="num_workers", param_type=ParamType.INT, default=0, description="Subprocesses for data loading", min_value=0),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        from torch.utils.data import DataLoader

        dataset = inputs["dataset"]
        batch_size = params.get("batch_size", 32)
        shuffle = params.get("shuffle", True)
        num_workers = params.get("num_workers", 0)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        return {"dataloader": dataloader}
