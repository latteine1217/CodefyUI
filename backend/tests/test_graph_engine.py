"""Tests for the graph engine: topological sort, validation, execution."""

import asyncio

import pytest

from app.core.graph_engine import (
    GraphValidationError,
    execute_graph,
    topological_sort,
    validate_graph,
)


def test_topological_sort_linear():
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c"},
    ]
    order = topological_sort(nodes, edges)
    assert order == ["a", "b", "c"]


def test_topological_sort_diamond():
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "a", "target": "c"},
        {"source": "b", "target": "d"},
        {"source": "c", "target": "d"},
    ]
    order = topological_sort(nodes, edges)
    assert order[0] == "a"
    assert order[-1] == "d"


def test_cycle_detection():
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c"},
        {"source": "c", "target": "a"},
    ]
    with pytest.raises(GraphValidationError):
        topological_sort(nodes, edges)


def test_validate_graph_valid():
    nodes = [
        {"id": "1", "type": "Print", "data": {"params": {}}},
        {"id": "2", "type": "Print", "data": {"params": {}}},
    ]
    edges = [
        {"source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
    ]
    errors = validate_graph(nodes, edges)
    assert errors == [], f"Unexpected errors: {errors}"


def test_validate_graph_unknown_node():
    nodes = [{"id": "1", "type": "NonExistentNode"}]
    edges = []
    # Should work (no edges to validate)
    errors = validate_graph(nodes, edges)
    assert errors == []


def test_validate_graph_type_mismatch():
    nodes = [
        {"id": "1", "type": "Loss"},
        {"id": "2", "type": "Conv2d"},
    ]
    edges = [
        {"source": "1", "target": "2", "sourceHandle": "loss_fn", "targetHandle": "tensor"},
    ]
    errors = validate_graph(nodes, edges)
    assert len(errors) > 0
    assert "mismatch" in errors[0].lower() or "Type" in errors[0]


@pytest.mark.asyncio
async def test_execute_print_nodes():
    nodes = [
        {"id": "1", "type": "Print", "data": {"params": {"label": "first"}}},
        {"id": "2", "type": "Print", "data": {"params": {"label": "second"}}},
    ]
    edges = [
        {"source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
    ]
    results = await execute_graph(nodes, edges)
    assert "1" in results
    assert "2" in results
