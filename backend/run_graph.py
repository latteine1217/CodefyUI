#!/usr/bin/env python3
"""
CodefyUI Graph CLI Runner
==========================
Execute a graph.json directly from the command line without starting the server.

Usage:
    python run_graph.py <path_to_graph.json>
    python run_graph.py ../examples/TrainCNN-MNIST/graph.json
    python run_graph.py ../examples/TrainCNN-MNIST/graph.json --validate-only
    python run_graph.py ../examples/TrainCNN-MNIST/graph.json --verbose
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.core.graph_engine import GraphValidationError, execute_graph, expand_presets, validate_graph
from app.core.node_registry import registry
from app.core.preset_registry import preset_registry


def _init_registries() -> None:
    """Discover all nodes and presets."""
    n = registry.discover(settings.NODES_DIR, "app.nodes")
    c = registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
    p = preset_registry.discover(settings.PRESETS_DIR, registry)
    print(f"[init] {n} built-in nodes, {c} custom nodes, {p} presets")


def _on_progress(node_id: str, status: str, data: dict[str, Any] | None) -> None:
    """CLI progress callback — prints node execution status."""
    if status == "running":
        print(f"  [{node_id}] running...")
    elif status == "completed":
        # Summarize outputs
        parts = []
        if data:
            for key, val in data.items():
                if hasattr(val, "shape"):
                    parts.append(f"{key}: Tensor{list(val.shape)}")
                elif hasattr(val, "parameters"):
                    n_params = sum(p.numel() for p in val.parameters())
                    parts.append(f"{key}: Model({n_params:,} params)")
                elif isinstance(val, (int, float)):
                    parts.append(f"{key}: {val}")
                elif isinstance(val, str) and len(val) > 80:
                    parts.append(f"{key}: str({len(val)} chars)")
                else:
                    parts.append(f"{key}: {type(val).__name__}")
        summary = ", ".join(parts) if parts else "ok"
        print(f"  [{node_id}] completed  ->  {summary}")
    elif status == "error":
        err = data.get("error", "unknown") if data else "unknown"
        print(f"  [{node_id}] ERROR: {err}")


async def run(graph_path: str, *, validate_only: bool = False, verbose: bool = False) -> None:
    t0 = time.time()

    # Load graph
    path = Path(graph_path)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    name = graph.get("name", path.stem)

    print(f"\n{'='*60}")
    print(f"  Graph: {name}")
    print(f"  File:  {path}")
    print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}")
    print(f"{'='*60}\n")

    # Expand presets
    expanded_nodes, expanded_edges, preset_map = expand_presets(nodes, edges)
    if len(expanded_nodes) != len(nodes):
        print(f"[presets] Expanded {len(nodes)} nodes -> {len(expanded_nodes)} (presets resolved)")

    # Validate
    print("[validate] Checking graph...")
    errors = validate_graph(expanded_nodes, expanded_edges)
    if errors:
        print("[validate] FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("[validate] OK\n")

    if validate_only:
        print("(--validate-only: skipping execution)")
        return

    # Execute
    print("[execute] Starting graph execution...")
    print("-" * 60)
    try:
        outputs = await execute_graph(
            nodes,
            edges,
            on_progress=_on_progress if verbose else _on_progress,
        )
    except GraphValidationError as e:
        print(f"\n[execute] Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[execute] Runtime error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    print("-" * 60)
    print(f"\n[done] Completed in {elapsed:.1f}s")

    # Print final summary
    print(f"\n{'='*60}")
    print("  Final Outputs")
    print(f"{'='*60}")
    for node_id, result in outputs.items():
        display_id = node_id
        # Shorten internal preset node IDs
        if "__" in node_id:
            preset_id, internal_id = node_id.split("__", 1)
            display_id = f"{preset_id}/{internal_id}"
        for key, val in result.items():
            if hasattr(val, "shape"):
                print(f"  {display_id}.{key} = Tensor{list(val.shape)}")
            elif hasattr(val, "parameters"):
                n_params = sum(p.numel() for p in val.parameters())
                print(f"  {display_id}.{key} = Model({n_params:,} params)")
            elif isinstance(val, (int, float)):
                print(f"  {display_id}.{key} = {val}")
            elif isinstance(val, str) and len(val) > 120:
                print(f"  {display_id}.{key} = str({len(val)} chars)")
            else:
                r = repr(val)
                if len(r) > 120:
                    r = r[:117] + "..."
                print(f"  {display_id}.{key} = {r}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute a CodefyUI graph.json from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("graph", help="Path to graph.json file")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, do not execute")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output and tracebacks")
    args = parser.parse_args()

    _init_registries()
    asyncio.run(run(args.graph, validate_only=args.validate_only, verbose=args.verbose))


if __name__ == "__main__":
    main()
