"""Tests for the type compatibility system."""

from app.core.node_base import DataType
from app.core.type_system import is_compatible


def test_same_type_compatible():
    for dt in DataType:
        assert is_compatible(dt, dt), f"{dt} should be compatible with itself"


def test_any_accepts_all():
    for dt in DataType:
        if dt is DataType.TRIGGER:
            continue
        assert is_compatible(dt, DataType.ANY), f"{dt} -> ANY should be compatible"


def test_any_source_to_any_target():
    for dt in DataType:
        if dt is DataType.TRIGGER:
            continue
        assert is_compatible(DataType.ANY, dt), f"ANY -> {dt} should be compatible"


def test_tensor_not_compatible_with_model():
    assert not is_compatible(DataType.TENSOR, DataType.MODEL)


def test_image_compatible_with_tensor():
    assert is_compatible(DataType.IMAGE, DataType.TENSOR)


def test_model_not_compatible_with_tensor():
    assert not is_compatible(DataType.MODEL, DataType.TENSOR)


def test_trigger_only_connects_to_trigger():
    assert is_compatible(DataType.TRIGGER, DataType.TRIGGER) is True


def test_trigger_does_not_connect_to_data():
    assert is_compatible(DataType.TRIGGER, DataType.TENSOR) is False
    assert is_compatible(DataType.TRIGGER, DataType.DATASET) is False
    assert is_compatible(DataType.TRIGGER, DataType.MODEL) is False


def test_data_does_not_connect_to_trigger():
    assert is_compatible(DataType.TENSOR, DataType.TRIGGER) is False
    assert is_compatible(DataType.DATASET, DataType.TRIGGER) is False


def test_any_does_not_swallow_trigger():
    """ANY is universal for data types but must NOT match TRIGGER, otherwise
    a trigger output could be connected to any data input."""
    assert is_compatible(DataType.ANY, DataType.TRIGGER) is False
    assert is_compatible(DataType.TRIGGER, DataType.ANY) is False
