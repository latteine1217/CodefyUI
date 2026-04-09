import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..config import settings

router = APIRouter(prefix="/api/examples", tags=["examples"])


@router.get("/list")
async def list_examples():
    base = settings.EXAMPLES_DIR
    if not base.exists():
        return []
    results = []
    for graph_file in sorted(base.rglob("graph.json")):
        try:
            data = json.loads(graph_file.read_text(encoding="utf-8"))
            rel = graph_file.parent.relative_to(base)
            parts = rel.parts
            category = parts[0] if parts else "Other"
            results.append({
                "name": data.get("name", rel.name),
                "description": data.get("description", ""),
                "category": category,
                "path": rel.as_posix(),
                "node_count": len(data.get("nodes", [])),
                "edge_count": len(data.get("edges", [])),
            })
        except Exception:
            continue
    return results


@router.get("/load")
async def load_example(path: str = Query(..., description="Relative path to the example directory")):
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path")
    resolved = (settings.EXAMPLES_DIR / path / "graph.json").resolve()
    if not str(resolved).startswith(str(settings.EXAMPLES_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Example not found: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))
