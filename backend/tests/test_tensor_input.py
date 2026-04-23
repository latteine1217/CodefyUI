"""Tests for TensorInput node."""

import pytest
import torch

from app.nodes.data.tensor_input_node import TensorInputNode


def test_explicit_mode_simple():
    node = TensorInputNode()
    result = node.execute(
        {},
        {
            "shape": "2,2",
            "dtype": "float32",
            "value_mode": "explicit",
            "values": [[1.0, 2.0], [3.0, 4.0]],
            "seed": 0,
        },
    )
    t = result["tensor"]
    assert t.shape == (2, 2)
    assert t.dtype == torch.float32
    assert torch.allclose(t, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_explicit_mode_flattens_and_reshapes():
    node = TensorInputNode()
    result = node.execute(
        {},
        {
            "shape": "1,2,3",
            "dtype": "float32",
            "value_mode": "explicit",
            "values": [1, 2, 3, 4, 5, 6],
            "seed": 0,
        },
    )
    t = result["tensor"]
    assert t.shape == (1, 2, 3)
    assert t[0, 1, 2].item() == 6.0


def test_explicit_mode_size_mismatch_raises():
    node = TensorInputNode()
    with pytest.raises(ValueError, match="elements"):
        node.execute(
            {},
            {
                "shape": "2,2",
                "dtype": "float32",
                "value_mode": "explicit",
                "values": [1, 2, 3],
                "seed": 0,
            },
        )


def test_random_mode_reproducible_with_seed():
    node = TensorInputNode()
    params = {"shape": "3,3", "dtype": "float32", "value_mode": "random", "values": None, "seed": 42}
    r1 = node.execute({}, params)["tensor"]
    r2 = node.execute({}, params)["tensor"]
    assert torch.allclose(r1, r2)


def test_zeros_mode():
    node = TensorInputNode()
    result = node.execute(
        {},
        {"shape": "2,3", "dtype": "float32", "value_mode": "zeros", "values": None, "seed": 0},
    )
    t = result["tensor"]
    assert t.shape == (2, 3)
    assert torch.all(t == 0)


def test_ones_mode():
    node = TensorInputNode()
    result = node.execute(
        {},
        {"shape": "2,3", "dtype": "float32", "value_mode": "ones", "values": None, "seed": 0},
    )
    t = result["tensor"]
    assert torch.all(t == 1)


def test_arange_mode():
    node = TensorInputNode()
    result = node.execute(
        {},
        {"shape": "2,3", "dtype": "float32", "value_mode": "arange", "values": None, "seed": 0},
    )
    t = result["tensor"]
    assert t.shape == (2, 3)
    assert t.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_dtype_int64():
    node = TensorInputNode()
    result = node.execute(
        {},
        {
            "shape": "2,2",
            "dtype": "int64",
            "value_mode": "explicit",
            "values": [[1, 2], [3, 4]],
            "seed": 0,
        },
    )
    assert result["tensor"].dtype == torch.int64


def test_dtype_bool():
    node = TensorInputNode()
    result = node.execute(
        {},
        {
            "shape": "2",
            "dtype": "bool",
            "value_mode": "explicit",
            "values": [1, 0],
            "seed": 0,
        },
    )
    t = result["tensor"]
    assert t.dtype == torch.bool
    assert t.tolist() == [True, False]


def test_shape_parsing_strips_whitespace():
    node = TensorInputNode()
    result = node.execute(
        {},
        {"shape": "1, 3, 4", "dtype": "float32", "value_mode": "zeros", "values": None, "seed": 0},
    )
    assert result["tensor"].shape == (1, 3, 4)


def test_empty_shape_raises():
    node = TensorInputNode()
    with pytest.raises(ValueError, match="shape"):
        node.execute(
            {},
            {"shape": "", "dtype": "float32", "value_mode": "zeros", "values": None, "seed": 0},
        )


def test_unsupported_dtype_raises():
    node = TensorInputNode()
    with pytest.raises(ValueError, match="dtype"):
        node.execute(
            {},
            {"shape": "2", "dtype": "float16", "value_mode": "zeros", "values": None, "seed": 0},
        )


def test_node_metadata():
    assert TensorInputNode.NODE_NAME == "TensorInput"
    assert TensorInputNode.CATEGORY == "Data"
    assert TensorInputNode.define_inputs() == []
    outputs = TensorInputNode.define_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "tensor"
