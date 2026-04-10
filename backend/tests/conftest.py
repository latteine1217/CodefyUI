"""Shared pytest fixtures for CodefyUI backend tests."""

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import NodeRegistry, registry
from app.core.preset_registry import preset_registry
from app.main import app


class _TestSourceNode(BaseNode):
    """Lightweight source node for tests -- no required inputs, no torch."""
    NODE_NAME = "_TestSource"
    CATEGORY = "Test"
    DESCRIPTION = "Emits a constant value"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"value": params.get("val", "test")}


@pytest.fixture(scope="session", autouse=True)
def registry_with_nodes() -> NodeRegistry:
    """Discover all nodes once per test session."""
    if len(registry.nodes) == 0:
        registry.discover(settings.NODES_DIR, "app.nodes")
        registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
        preset_registry.discover(settings.PRESETS_DIR, registry)
    registry._nodes["_TestSource"] = _TestSourceNode
    return registry


@pytest.fixture
async def test_client():
    """Async HTTP client connected to the FastAPI app via ASGI transport."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_graph():
    """A minimal valid graph: Start -> _TestSource -> Print."""
    return {
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": -150, "y": 0}, "data": {"params": {}}},
            {"id": "1", "type": "_TestSource", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {"id": "2", "type": "Print", "position": {"x": 200, "y": 0}, "data": {"params": {"label": "second"}}},
        ],
        "edges": [
            {"id": "et", "source": "start", "target": "1", "sourceHandle": "trigger", "type": "trigger"},
            {"id": "e1", "source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
        ],
        "name": "test-graph",
        "description": "A test graph",
    }
