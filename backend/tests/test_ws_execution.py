"""Tests for WebSocket execution endpoint."""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from app.main import app


@pytest.mark.asyncio
async def test_ws_connect_and_execute():
    """Test that we can connect via WS and execute a simple graph."""
    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url="http://test",
    ) as client:
        async with aconnect_ws("/ws/execution", client) as ws:
            # Send execute with two Print nodes
            await ws.send_text(json.dumps({
                "action": "execute",
                "nodes": [
                    {"id": "1", "type": "Print", "data": {"params": {"label": "a"}}},
                    {"id": "2", "type": "Print", "data": {"params": {"label": "b"}}},
                ],
                "edges": [
                    {"source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
                ],
            }))

            # Collect messages until execution_complete or error
            messages = []
            for _ in range(20):
                msg = json.loads(await ws.receive_text())
                messages.append(msg)
                if msg["type"] in ("execution_complete", "execution_error"):
                    break

            types = [m["type"] for m in messages]
            assert "execution_start" in types
            assert "execution_complete" in types


@pytest.mark.asyncio
async def test_ws_unknown_action():
    """Unknown actions should return an error message."""
    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url="http://test",
    ) as client:
        async with aconnect_ws("/ws/execution", client) as ws:
            await ws.send_text(json.dumps({"action": "foobar"}))
            msg = json.loads(await ws.receive_text())
            assert msg["type"] == "error"
            assert "foobar" in msg["error"]
