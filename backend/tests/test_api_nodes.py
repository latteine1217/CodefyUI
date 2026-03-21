"""Tests for the nodes API endpoints."""

import pytest


@pytest.mark.asyncio
async def test_list_nodes(test_client):
    resp = await test_client.get("/api/nodes")
    assert resp.status_code == 200
    nodes = resp.json()
    assert isinstance(nodes, list)
    assert len(nodes) >= 1
    # Check structure of first node
    node = nodes[0]
    assert "node_name" in node
    assert "category" in node
    assert "inputs" in node
    assert "outputs" in node
    assert "params" in node


@pytest.mark.asyncio
async def test_get_specific_node(test_client):
    resp = await test_client.get("/api/nodes/Conv2d")
    assert resp.status_code == 200
    node = resp.json()
    assert node["node_name"] == "Conv2d"
    assert node["category"] == "CNN"


@pytest.mark.asyncio
async def test_get_nonexistent_node(test_client):
    resp = await test_client.get("/api/nodes/DoesNotExist")
    assert resp.status_code == 404
