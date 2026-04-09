# backend/tests/test_graph_model.py
import json

import pytest
import torch

from app.nodes.utility.graph_model import build_graph_model


def _spec(nodes, edges):
    return {"version": 2, "nodes": nodes, "edges": edges}


def test_linear_dag_single_layer():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_in", "name": "x"}]},
            {"id": "lin", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_out", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_in", "target": "lin", "targetHandle": None},
            {"id": "e2", "source": "lin", "sourceHandle": None, "target": "out", "targetHandle": "p_out"},
        ],
    )

    model = build_graph_model(spec)
    x = torch.randn(3, 4)
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 2)


def test_residual_block_with_add():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_in", "name": "x"}]},
            {"id": "lin1", "type": "Linear", "params": {"in_features": 4, "out_features": 4}},
            {"id": "relu", "type": "ReLU"},
            {"id": "lin2", "type": "Linear", "params": {"in_features": 4, "out_features": 4}},
            {"id": "add", "type": "Add"},
            {"id": "out", "type": "Output", "ports": [{"id": "p_out", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_in", "target": "lin1"},
            {"id": "e2", "source": "lin1", "target": "relu"},
            {"id": "e3", "source": "relu", "target": "lin2"},
            {"id": "e4", "source": "lin2", "target": "add"},
            {"id": "e5", "source": "in", "sourceHandle": "p_in", "target": "add"},  # skip
            {"id": "e6", "source": "add", "target": "out", "targetHandle": "p_out"},
        ],
    )
    model = build_graph_model(spec)
    x = torch.randn(2, 4)
    y = model(x)
    assert y.shape == (2, 4)
    # Skip means at least one parameter pathway exists; verify gradient flows
    y.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
