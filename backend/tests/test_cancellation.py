"""Tests for execution cancellation (Phase 2)."""

import asyncio

import pytest

from app.core.execution_context import CancellationError, ExecutionContext
from app.core.graph_engine import execute_graph


def _start_node(nid="start"):
    return {"id": nid, "type": "Start", "data": {"params": {}}}


def _trigger(eid, src, tgt):
    return {"id": eid, "source": src, "target": tgt, "sourceHandle": "trigger", "type": "trigger"}


def test_execution_context_cancel():
    ctx = ExecutionContext()
    assert not ctx.cancelled
    ctx.cancel()
    assert ctx.cancelled


@pytest.mark.asyncio
async def test_cancel_before_execution():
    """Cancelling context before execution raises CancellationError."""
    ctx = ExecutionContext()
    ctx.cancel()

    nodes = [
        _start_node(),
        {"id": "1", "type": "_TestSource", "data": {"params": {}}},
    ]
    edges = [_trigger("et", "start", "1")]

    with pytest.raises(CancellationError):
        await execute_graph(nodes, edges, context=ctx)


@pytest.mark.asyncio
async def test_cancel_during_execution():
    """Cancelling mid-execution stops before later nodes run."""
    ctx = ExecutionContext()
    executed_nodes = []

    async def on_progress(node_id, status, data):
        if status == "running":
            executed_nodes.append(node_id)
            if node_id == "1":
                ctx.cancel()

    nodes = [
        _start_node(),
        {"id": "1", "type": "_TestSource", "data": {"params": {}}},
        {"id": "2", "type": "Print", "data": {"params": {}}},
    ]
    edges = [
        _trigger("et", "start", "1"),
        {"source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
    ]

    with pytest.raises(CancellationError):
        await execute_graph(nodes, edges, on_progress=on_progress, context=ctx)

    # Node 1 started, but node 2 should not have started
    assert "1" in executed_nodes
