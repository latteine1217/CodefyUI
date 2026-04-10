"""One-shot migration: add `isEntryPoint: true` to data-root nodes in
all example graphs.

Run from the repository root:

    python scripts/migrate_entry_points.py

This script is idempotent — running it twice has no further effect.
"""

import json
import sys
from pathlib import Path


def migrate_graph(graph_path: Path) -> bool:
    """Returns True if the file was modified."""
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # Find data-roots: nodes that are NOT the target of any DATA edge.
    # (Trigger edges don't count, but example graphs have no trigger
    # edges yet — they all use data edges.)
    targets = {
        e["target"]
        for e in edges
        if e.get("type", "data") == "data"
    }
    roots = [n for n in nodes if n["id"] not in targets]

    modified = False
    for root in roots:
        node_data = root.setdefault("data", {})
        if not node_data.get("isEntryPoint"):
            node_data["isEntryPoint"] = True
            modified = True

    if modified:
        graph_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return modified


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    examples = sorted((repo_root / "examples").rglob("graph.json"))
    if not examples:
        print("No graph.json files found under examples/")
        return 1

    total = 0
    modified = 0
    for path in examples:
        rel = path.relative_to(repo_root)
        was_modified = migrate_graph(path)
        total += 1
        marker = "MODIFIED" if was_modified else "skip    "
        if was_modified:
            modified += 1
        print(f"  {marker} {rel}")

    print(f"\nDone: {modified}/{total} graphs updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
