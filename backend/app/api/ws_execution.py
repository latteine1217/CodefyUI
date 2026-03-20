import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core.graph_engine import GraphValidationError, execute_graph

router = APIRouter()


@router.websocket("/ws/execution")
async def websocket_execution(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            action = data.get("action")
            if action == "execute":
                nodes = data.get("nodes", [])
                edges = data.get("edges", [])

                async def on_progress(node_id: str, status: str, result: dict[str, Any] | None) -> None:
                    msg: dict[str, Any] = {
                        "type": "node_status",
                        "node_id": node_id,
                        "status": status,
                    }
                    if result and status == "error":
                        msg["error"] = result.get("error", "")
                    if result and status == "completed":
                        # Forward log output (from Print node etc.)
                        if "__log__" in result:
                            msg["log"] = str(result["__log__"])
                        # Forward base64 image data so the frontend can display it
                        for key, val in result.items():
                            if key.startswith("__"):
                                continue
                            if isinstance(val, str) and len(val) > 200 and val[:20].isalnum():
                                msg["image"] = val
                                break
                    await ws.send_text(json.dumps(msg))

                try:
                    await ws.send_text(json.dumps({"type": "execution_start"}))
                    await execute_graph(nodes, edges, on_progress=on_progress)
                    await ws.send_text(json.dumps({"type": "execution_complete"}))
                except GraphValidationError as e:
                    await ws.send_text(json.dumps({"type": "execution_error", "error": str(e)}))
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "execution_error", "error": str(e)}))
            elif action == "stop":
                await ws.send_text(json.dumps({"type": "execution_stopped"}))
            else:
                await ws.send_text(json.dumps({"type": "error", "error": f"Unknown action: {action}"}))
    except WebSocketDisconnect:
        pass
