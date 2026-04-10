"""Tests for the graph engine: topological sort, validation, execution."""

import asyncio

import pytest

from app.core.graph_engine import (
    GraphValidationError,
    execute_graph,
    topological_levels,
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


def _start_node(nid="start"):
    return {"id": nid, "type": "Start", "data": {"params": {}}}


def _trigger(eid, src, tgt):
    return {"id": eid, "source": src, "target": tgt, "sourceHandle": "trigger", "type": "trigger"}


def test_validate_graph_valid():
    """_TestSource has no required inputs, Print's required input is satisfied by the edge."""
    nodes = [
        _start_node(),
        {"id": "1", "type": "_TestSource", "data": {"params": {}}},
        {"id": "2", "type": "Print", "data": {"params": {}}},
    ]
    edges = [
        _trigger("et", "start", "1"),
        {"source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
    ]
    errors = validate_graph(nodes, edges)
    assert errors == [], f"Unexpected errors: {errors}"


def test_validate_graph_unknown_node():
    nodes = [
        _start_node(),
        {"id": "1", "type": "NonExistentNode", "data": {}},
    ]
    edges = [_trigger("et", "start", "1")]
    errors = validate_graph(nodes, edges)
    assert any("Unknown node type" in e and "NonExistentNode" in e and "node 1" in e for e in errors)


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
    """Use _TestSource (registered in conftest, no torch) to feed Print."""
    nodes = [
        _start_node(),
        {"id": "1", "type": "_TestSource", "data": {"params": {}}},
        {"id": "2", "type": "Print", "data": {"params": {"label": "second"}}},
    ]
    edges = [
        _trigger("et", "start", "1"),
        {"source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
    ]
    results = await execute_graph(nodes, edges)
    assert "1" in results
    assert "2" in results


def test_validate_graph_missing_required_input():
    """Conv2d has a required 'tensor' input; without an edge it should error."""
    nodes = [
        {"id": "1", "type": "Conv2d", "data": {"params": {}}},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    assert any("Missing required input" in e and "'tensor'" in e and "node 1" in e for e in errors)


def test_validate_graph_optional_input_no_error():
    """TrainingLoop has optional inputs (val_dataloader, lr_scheduler); leaving them
    unconnected should not produce errors for those ports."""
    nodes = [
        {"id": "1", "type": "TrainingLoop", "data": {"params": {}}},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    # Should have errors for the required inputs but NOT for the optional ones
    optional_names = {"val_dataloader", "lr_scheduler"}
    for e in errors:
        for name in optional_names:
            assert name not in e, f"Optional input '{name}' should not cause an error"


def test_validate_graph_param_below_min():
    """Dropout 'p' has min_value=0.0; supplying -0.5 should error."""
    nodes = [
        {"id": "1", "type": "Dropout", "data": {"params": {"p": -0.5}}},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    assert any("below minimum" in e and "'p'" in e for e in errors)


def test_validate_graph_param_above_max():
    """Dropout 'p' has max_value=1.0; supplying 1.5 should error."""
    nodes = [
        {"id": "1", "type": "Dropout", "data": {"params": {"p": 1.5}}},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    assert any("above maximum" in e and "'p'" in e for e in errors)


def test_validate_graph_param_within_range_no_error():
    """Dropout 'p' within [0.0, 1.0] should not produce a range error."""
    nodes = [
        {"id": "1", "type": "Dropout", "data": {"params": {"p": 0.5}}},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    range_errors = [e for e in errors if "below minimum" in e or "above maximum" in e]
    assert range_errors == []


def test_validate_graph_multiple_unknown_nodes():
    """Multiple unknown node types should each produce their own error."""
    nodes = [
        {"id": "1", "type": "FakeNodeA"},
        {"id": "2", "type": "FakeNodeB"},
    ]
    edges = []
    errors = validate_graph(nodes, edges)
    assert len([e for e in errors if "Unknown node type" in e]) == 2
    assert any("FakeNodeA" in e for e in errors)
    assert any("FakeNodeB" in e for e in errors)


def test_validate_graph_required_input_satisfied_by_edge():
    """Conv2d's required 'tensor' input connected via an edge should pass the check."""
    nodes = [
        {"id": "1", "type": "Conv2d", "data": {"params": {}}},
        {"id": "2", "type": "Conv2d", "data": {"params": {}}},
    ]
    edges = [
        {"source": "1", "target": "2", "sourceHandle": "tensor", "targetHandle": "tensor"},
    ]
    errors = validate_graph(nodes, edges)
    # Node 1 still has a missing required input, but node 2's 'tensor' is satisfied
    node2_missing = [e for e in errors if "node 2" in e and "Missing required input" in e]
    assert node2_missing == [], f"Node 2's input should be satisfied: {node2_missing}"


from app.core.graph_engine import find_entry_points, reachable_from_entry_points


def test_find_entry_points_only_trigger_targets():
    """Only nodes with incoming trigger edges are entry points."""
    nodes = [
        _start_node(),
        {"id": "ds", "type": "Dataset", "data": {}},
        {"id": "other", "type": "Dataset", "data": {}},
    ]
    edges = [_trigger("e1", "start", "ds")]
    result = find_entry_points(nodes, edges)
    assert "ds" in result
    assert "start" not in result  # Start itself is NOT an entry point
    assert "other" not in result


def test_find_entry_points_via_trigger_edge():
    nodes = [
        _start_node(),
        {"id": "ds", "type": "Dataset", "data": {}},
    ]
    edges = [_trigger("e1", "start", "ds")]
    result = find_entry_points(nodes, edges)
    assert result == ["ds"]
    assert "start" not in result


def test_find_entry_points_multiple_targets():
    nodes = [
        _start_node(),
        {"id": "a", "type": "Dataset", "data": {}},
        {"id": "b", "type": "Dataset", "data": {}},
    ]
    edges = [
        _trigger("e1", "start", "a"),
        _trigger("e2", "start", "b"),
    ]
    result = set(find_entry_points(nodes, edges))
    assert result == {"a", "b"}


def test_find_entry_points_none():
    nodes = [{"id": "a"}, {"id": "b"}]
    edges = [{"id": "e1", "source": "a", "target": "b", "type": "data"}]
    assert find_entry_points(nodes, edges) == []


def test_reachable_traverses_data_edges_only():
    nodes = [{"id": n} for n in ["start", "ds", "dl", "model"]]
    edges = [
        {"id": "e1", "source": "start", "target": "ds", "type": "trigger"},
        {"id": "e2", "source": "ds", "target": "dl", "type": "data"},
        {"id": "e3", "source": "dl", "target": "model", "type": "data"},
    ]
    reachable = reachable_from_entry_points(["ds"], edges)
    assert reachable == {"ds", "dl", "model"}


def test_reachable_handles_disconnected_components():
    nodes = [{"id": n} for n in ["a", "b", "x", "y"]]
    edges = [
        {"id": "e1", "source": "a", "target": "b", "type": "data"},
        {"id": "e2", "source": "x", "target": "y", "type": "data"},
    ]
    assert reachable_from_entry_points(["a"], edges) == {"a", "b"}


def _make_node(nid, ntype="Dataset", is_entry=False):
    return {"id": nid, "type": ntype, "data": {"params": {}}}


def _make_edge(eid, src, tgt, etype="data"):
    return {
        "id": eid,
        "source": src,
        "target": tgt,
        "sourceHandle": "out",
        "targetHandle": "in",
        "type": etype,
    }


def test_validate_rejects_no_entry_points():
    nodes = [_make_node("a"), _make_node("b")]
    edges = [_make_edge("e1", "a", "b")]
    errors = validate_graph(nodes, edges)
    assert any("entry point" in err.lower() for err in errors)


def test_validate_accepts_start_with_trigger():
    nodes = [_start_node(), _make_node("a"), _make_node("b")]
    edges = [
        _trigger("et", "start", "a"),
        _make_edge("e1", "a", "b"),
    ]
    errors = validate_graph(nodes, edges)
    assert not any("entry point" in err.lower() for err in errors)


def test_validate_allows_trigger_to_any_node():
    """Start trigger can connect to any node — it's a control-flow marker, not data."""
    nodes = [
        _start_node(),
        {"id": "conv", "type": "Conv2d", "data": {"params": {}}},
    ]
    edges = [_trigger("et", "start", "conv")]
    errors = validate_graph(nodes, edges)
    # Conv2d will have a "missing required input" error (its tensor input),
    # but NOT a "trigger cannot connect" error.
    assert not any("trigger" in err.lower() for err in errors)


def test_validate_allows_cycle_in_draft_component():
    """A cycle inside a non-entry-pointed (draft) component should NOT
    fail validation, because the draft is skipped at execution."""
    nodes = [
        _start_node(),
        _make_node("ep"),
        _make_node("a"),
        _make_node("b"),
        _make_node("c"),
    ]
    edges = [
        _trigger("et", "start", "ep"),
        # Cycle in the draft component a->b->c->a
        _make_edge("e1", "a", "b"),
        _make_edge("e2", "b", "c"),
        _make_edge("e3", "c", "a"),
    ]
    errors = validate_graph(nodes, edges)
    assert not any("cycle" in err.lower() for err in errors)


def test_validate_rejects_cycle_in_entry_pointed_component():
    """A cycle in an entry-pointed component fails."""
    nodes = [
        _start_node(),
        _make_node("a"),
        _make_node("b"),
    ]
    edges = [
        _trigger("et", "start", "a"),
        _make_edge("e1", "a", "b"),
        _make_edge("e2", "b", "a"),
    ]
    errors = validate_graph(nodes, edges)
    assert any("cycle" in err.lower() for err in errors)


def test_topological_levels_excludes_trigger_from_in_degree():
    """A Dataset receiving a trigger from a Start should still be at level 0."""
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "ds", "type": "Dataset", "data": {"params": {}}},
        {"id": "dl", "type": "DataLoader", "data": {"params": {}}},
    ]
    edges = [
        {"id": "e1", "source": "start", "target": "ds", "sourceHandle": "trigger", "type": "trigger"},
        {"id": "e2", "source": "ds", "target": "dl", "sourceHandle": "dataset", "targetHandle": "dataset", "type": "data"},
    ]
    levels = topological_levels(nodes, edges)
    # Both start and ds should be in level 0 (start has no inputs, ds's
    # only incoming edge is a trigger which is excluded).
    assert "start" in levels[0]
    assert "ds" in levels[0]
    assert "dl" in levels[1]


@pytest.mark.asyncio
async def test_execute_graph_skips_draft_components():
    """Draft components (no Start trigger) should be skipped silently."""
    nodes = [
        _start_node(),
        {"id": "live", "type": "_TestSource", "data": {"params": {"val": 42}}},
        {"id": "draft", "type": "_TestSource", "data": {"params": {"val": 99}}},
    ]
    edges = [_trigger("et", "start", "live")]
    results = await execute_graph(nodes, edges)
    assert "live" in results
    assert "draft" not in results
