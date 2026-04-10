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


def test_validate_graph_valid():
    """_TestSource has no required inputs, Print's required input is satisfied by the edge."""
    nodes = [
        {"id": "1", "type": "_TestSource", "data": {"params": {}, "isEntryPoint": True}},
        {"id": "2", "type": "Print", "data": {"params": {}}},
    ]
    edges = [
        {"source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
    ]
    errors = validate_graph(nodes, edges)
    assert errors == [], f"Unexpected errors: {errors}"


def test_validate_graph_unknown_node():
    nodes = [{"id": "1", "type": "NonExistentNode", "data": {"isEntryPoint": True}}]
    edges = []
    errors = validate_graph(nodes, edges)
    assert len(errors) == 1
    assert "Unknown node type" in errors[0]
    assert "NonExistentNode" in errors[0]
    assert "node 1" in errors[0]


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
        {"id": "1", "type": "_TestSource", "data": {"params": {}, "isEntryPoint": True}},
        {"id": "2", "type": "Print", "data": {"params": {"label": "second"}}},
    ]
    edges = [
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


def test_find_entry_points_explicit_marker():
    nodes = [
        {"id": "a", "data": {"isEntryPoint": True}},
        {"id": "b", "data": {"isEntryPoint": False}},
        {"id": "c", "data": {"isEntryPoint": False}},
    ]
    edges = []
    assert find_entry_points(nodes, edges) == ["a"]


def test_find_entry_points_via_trigger_edge():
    nodes = [
        {"id": "start", "type": "Start", "data": {"isEntryPoint": False}},
        {"id": "ds", "type": "Dataset", "data": {"isEntryPoint": False}},
    ]
    edges = [
        {"id": "e1", "source": "start", "target": "ds", "type": "trigger"},
    ]
    # The DOWNSTREAM node (ds) is the entry point because it has an
    # incoming trigger edge.
    assert "ds" in find_entry_points(nodes, edges)
    # Start node itself is also an entry (it has isEntryPoint by virtue
    # of being a Start? — no: Start nodes are entry points because they're
    # always treated as such; we'll handle that via data.isEntryPoint=True
    # being set when the StartNode is instantiated on the canvas, OR by
    # treating Start type as implicit entry. We'll go with implicit-by-type
    # below.)
    assert "start" in find_entry_points(nodes, edges)


def test_find_entry_points_combined():
    nodes = [
        {"id": "a", "type": "Dataset", "data": {"isEntryPoint": True}},
        {"id": "start", "type": "Start", "data": {"isEntryPoint": False}},
        {"id": "b", "type": "Dataset", "data": {"isEntryPoint": False}},
        {"id": "c", "type": "Conv", "data": {"isEntryPoint": False}},
    ]
    edges = [
        {"id": "e1", "source": "start", "target": "b", "type": "trigger"},
        {"id": "e2", "source": "a", "target": "c", "type": "data"},
    ]
    result = set(find_entry_points(nodes, edges))
    assert result == {"a", "b", "start"}


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
    # Starting from `ds` (data root that received a trigger), BFS through
    # data edges should reach ds, dl, model — but NOT start (it's upstream
    # via a trigger edge, which is not traversed).
    reachable = reachable_from_entry_points(["ds"], edges)
    assert reachable == {"ds", "dl", "model"}


def test_reachable_includes_start_when_explicitly_in_seed():
    nodes = [{"id": n} for n in ["start", "ds"]]
    edges = [{"id": "e1", "source": "start", "target": "ds", "type": "trigger"}]
    # If we seed BFS with start AND ds, both are in the result; trigger
    # edges are still not traversed, but the seeds themselves are included.
    reachable = reachable_from_entry_points(["start", "ds"], edges)
    assert reachable == {"start", "ds"}


def test_reachable_handles_disconnected_components():
    nodes = [{"id": n} for n in ["a", "b", "x", "y"]]
    edges = [
        {"id": "e1", "source": "a", "target": "b", "type": "data"},
        {"id": "e2", "source": "x", "target": "y", "type": "data"},
    ]
    assert reachable_from_entry_points(["a"], edges) == {"a", "b"}


def _make_node(nid, ntype="Dataset", is_entry=False):
    return {
        "id": nid,
        "type": ntype,
        "data": {"params": {}, "isEntryPoint": is_entry},
    }


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


def test_validate_accepts_single_entry_point():
    nodes = [_make_node("a", is_entry=True), _make_node("b")]
    edges = [_make_edge("e1", "a", "b")]
    errors = validate_graph(nodes, edges)
    # Should pass entry-point check (other validation may still complain
    # about node type registration; we only care entry-point rule passes)
    assert not any("entry point" in err.lower() for err in errors)


def test_validate_rejects_entry_with_incoming_data_edge():
    """A node marked data.isEntryPoint=True must not have incoming data edges."""
    nodes = [_make_node("a"), _make_node("b", is_entry=True)]
    edges = [_make_edge("e1", "a", "b", etype="data")]
    errors = validate_graph(nodes, edges)
    assert any("data-root" in err.lower() or "incoming" in err.lower() for err in errors)


def test_validate_rejects_trigger_target_with_incoming_data_edge():
    """A node that is an entry via incoming trigger ALSO must be a data-root."""
    nodes = [
        _make_node("start", ntype="Start"),
        _make_node("upstream"),
        _make_node("target"),
    ]
    edges = [
        _make_edge("e1", "start", "target", etype="trigger"),
        _make_edge("e2", "upstream", "target", etype="data"),
    ]
    errors = validate_graph(nodes, edges)
    assert any("data-root" in err.lower() or "incoming" in err.lower() for err in errors)


def test_validate_allows_cycle_in_draft_component():
    """A cycle inside a non-entry-pointed (draft) component should NOT
    fail validation, because the draft is skipped at execution."""
    nodes = [
        _make_node("ep", is_entry=True),
        _make_node("a"),
        _make_node("b"),
        _make_node("c"),
    ]
    edges = [
        # Cycle in the draft component a->b->c->a
        _make_edge("e1", "a", "b"),
        _make_edge("e2", "b", "c"),
        _make_edge("e3", "c", "a"),
    ]
    errors = validate_graph(nodes, edges)
    assert not any("cycle" in err.lower() for err in errors)


def test_validate_rejects_cycle_in_entry_pointed_component():
    """A cycle in an entry-pointed component fails (current behaviour)."""
    nodes = [
        _make_node("ep", is_entry=True),
        _make_node("a"),
        _make_node("b"),
    ]
    edges = [
        _make_edge("e1", "ep", "a"),
        _make_edge("e2", "a", "b"),
        _make_edge("e3", "b", "ep"),  # cycle back
    ]
    errors = validate_graph(nodes, edges)
    # Note: this will also fail rule "entry must be data-root" because ep
    # has an incoming edge from b. So we accept either error here.
    assert any(("cycle" in err.lower()) or ("data-root" in err.lower()) for err in errors)


def test_topological_levels_excludes_trigger_from_in_degree():
    """A Dataset receiving a trigger from a Start should still be at level 0."""
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}, "isEntryPoint": False}},
        {"id": "ds", "type": "Dataset", "data": {"params": {}, "isEntryPoint": False}},
        {"id": "dl", "type": "DataLoader", "data": {"params": {}, "isEntryPoint": False}},
    ]
    edges = [
        {"id": "e1", "source": "start", "target": "ds", "sourceHandle": "trigger", "targetHandle": "trigger", "type": "trigger"},
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
    """Draft components (no entry point) should be skipped silently."""
    # We need real registered nodes for execute_graph to actually run.
    # _TestSource is registered in conftest.py and takes no inputs.
    nodes = [
        {"id": "live", "type": "_TestSource", "data": {"params": {"val": 42}, "isEntryPoint": True}},
        {"id": "draft", "type": "_TestSource", "data": {"params": {"val": 99}, "isEntryPoint": False}},
    ]
    edges = []
    results = await execute_graph(nodes, edges)
    assert "live" in results
    assert "draft" not in results
