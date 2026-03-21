"""Shared pytest fixtures for CodefyUI backend tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.node_registry import NodeRegistry, registry
from app.core.preset_registry import preset_registry
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def registry_with_nodes() -> NodeRegistry:
    """Discover all nodes once per test session."""
    if len(registry.nodes) == 0:
        registry.discover(settings.NODES_DIR, "app.nodes")
        registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
        preset_registry.discover(settings.PRESETS_DIR, registry)
    return registry


@pytest.fixture
async def test_client():
    """Async HTTP client connected to the FastAPI app via ASGI transport."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_graph():
    """A minimal valid graph with two Print nodes."""
    return {
        "nodes": [
            {"id": "1", "type": "Print", "position": {"x": 0, "y": 0}, "data": {"params": {"label": "first"}}},
            {"id": "2", "type": "Print", "position": {"x": 200, "y": 0}, "data": {"params": {"label": "second"}}},
        ],
        "edges": [
            {"id": "e1", "source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
        ],
        "name": "test-graph",
        "description": "A test graph",
    }
