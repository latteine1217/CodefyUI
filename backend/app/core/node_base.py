from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DataType(str, Enum):
    TENSOR = "TENSOR"
    MODEL = "MODEL"
    DATASET = "DATASET"
    DATALOADER = "DATALOADER"
    OPTIMIZER = "OPTIMIZER"
    LOSS_FN = "LOSS_FN"
    SCALAR = "SCALAR"
    STRING = "STRING"
    IMAGE = "IMAGE"
    LIST = "LIST"
    ANY = "ANY"
    TRIGGER = "TRIGGER"


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    SELECT = "select"
    MODEL_FILE = "model_file"


@dataclass
class PortDefinition:
    name: str
    data_type: DataType
    description: str = ""
    optional: bool = False


@dataclass
class ParamDefinition:
    name: str
    param_type: ParamType
    default: Any = None
    description: str = ""
    options: list[str] = field(default_factory=list)  # for SELECT type
    min_value: float | None = None
    max_value: float | None = None


class BaseNode(ABC):
    NODE_NAME: str = ""
    CATEGORY: str = ""
    DESCRIPTION: str = ""

    @classmethod
    @abstractmethod
    def define_inputs(cls) -> list[PortDefinition]:
        ...

    @classmethod
    @abstractmethod
    def define_outputs(cls) -> list[PortDefinition]:
        ...

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return []

    @abstractmethod
    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        ...
