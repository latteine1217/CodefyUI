from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import routes_graph, routes_nodes, routes_presets, ws_execution
from .config import settings
from .core.node_registry import registry
from .core.preset_registry import preset_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Discover built-in nodes
    count = registry.discover(settings.NODES_DIR, "app.nodes")
    print(f"[CodefyUI] Discovered {count} built-in nodes")

    # Discover custom nodes
    custom_count = registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
    print(f"[CodefyUI] Discovered {custom_count} custom nodes")

    for name in sorted(registry.nodes.keys()):
        print(f"  - {name} ({registry.nodes[name].CATEGORY})")

    # Discover presets
    preset_count = preset_registry.discover(settings.PRESETS_DIR, registry)
    print(f"[CodefyUI] Discovered {preset_count} presets")
    for name in sorted(preset_registry.presets.keys()):
        print(f"  * {name}")

    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_nodes.router)
app.include_router(routes_graph.router)
app.include_router(routes_presets.router)
app.include_router(ws_execution.router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "nodes_loaded": len(registry.nodes),
        "presets_loaded": len(preset_registry.presets),
    }


@app.post("/api/nodes/reload")
async def reload_nodes():
    registry.clear()
    count = registry.discover(settings.NODES_DIR, "app.nodes")
    custom_count = registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
    preset_registry.clear()
    preset_count = preset_registry.discover(settings.PRESETS_DIR, registry)
    return {
        "builtin": count,
        "custom": custom_count,
        "presets": preset_count,
        "total": count + custom_count,
    }
