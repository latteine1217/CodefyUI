# Entry Points + Auto Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit Entry Point markers (inline + dedicated `Start` node type, Scratch hat-block style) so users can declare which subgraphs run when multiple sit on the canvas, plus an Auto Layout feature using Dagre with swim-lane packing per connected component.

**Architecture:** Backend gains a `DataType.TRIGGER` value, an `isEntryPoint` flag on `NodeData`, a `type` discriminator on `EdgeData` (`"data"` | `"trigger"`), and a no-op `StartNode`. `validate_graph()` enforces entry-point rules; `execute_graph()` filters to nodes reachable from entry points (drafts are skipped silently); in-degree calculations exclude trigger edges. Frontend renders entry-point nodes with a green outline + flag badge, ships a small `ContextMenu` primitive for right-click marking, integrates Dagre for LR layered layout with swim lanes per component, and a Migration Modal handles loading old graphs without entry points.

**Tech Stack:** Backend — Python 3.11, FastAPI, Pydantic, pytest. Frontend — React 19, React Flow v12 (`@xyflow/react`), Zustand, Vite, Vitest, `@dagrejs/dagre` (new). Manual verification via `mcp__claude-in-chrome__*` MCP tools.

**Spec:** [docs/superpowers/specs/2026-04-10-entry-points-auto-layout-design.md](../specs/2026-04-10-entry-points-auto-layout-design.md)

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `backend/app/core/node_base.py` | modify | Add `DataType.TRIGGER` enum value |
| `backend/app/core/type_system.py` | modify | Add `TRIGGER → {TRIGGER}` to `_COMPAT`; ensure `ANY` does NOT include `TRIGGER` |
| `backend/app/schemas/models.py` | modify | Add `NodeData.isEntryPoint: bool = False`; add `EdgeData.type: Literal["data","trigger"] = "data"` |
| `backend/app/nodes/control/__init__.py` | create | Empty package marker (auto-discovery walks the directory) |
| `backend/app/nodes/control/start_node.py` | create | `StartNode(BaseNode)` — no inputs, one trigger output, `execute()` returns `{}` |
| `backend/app/core/graph_engine.py` | modify | Add `find_entry_points()` and `reachable_from_entry_points()`; gate `validate_graph`/`execute_graph` on reachability; exclude trigger edges from in-degree |
| `backend/tests/test_graph_engine.py` | modify | Add 9 new test cases for entry-point semantics |
| `frontend/package.json` | modify | Add `@dagrejs/dagre` dependency |
| `frontend/src/types/index.ts` (or wherever NodeData/Edge types live) | modify | Add `isEntryPoint?: boolean` to NodeData; add `type?: 'data' \| 'trigger'` to edge data |
| `frontend/src/components/Nodes/flag-icon.svg` | create | 16×16 inline SVG flag icon (uses `currentColor`) |
| `frontend/src/components/Nodes/BaseNode.module.css` | modify | Add `.entryPoint` modifier (green outline + flag badge via `::before`) |
| `frontend/src/components/Nodes/BaseNode.tsx` | modify | Apply `.entryPoint` modifier conditionally; wire right-click → `ContextMenu` |
| `frontend/src/components/Nodes/StartNode.tsx` | create | Pill-shaped node, single trigger output, always entry-point styling |
| `frontend/src/components/Nodes/StartNode.module.css` | create | Pill styling for StartNode |
| `frontend/src/components/shared/ContextMenu.tsx` | create | Generic context menu primitive (positioned at cursor, click-outside dismissal) |
| `frontend/src/components/shared/ContextMenu.module.css` | create | Context menu styles |
| `frontend/src/components/Canvas/FlowCanvas.tsx` | modify | Register `start` in `nodeTypes`; register `triggerEdge` in `edgeTypes`; mark new edges from trigger handles as `triggerEdge` type |
| `frontend/src/components/Canvas/TriggerEdge.tsx` | create | Custom React Flow edge component (green dashed bezier) |
| `frontend/src/components/Sidebar/NodePalette.tsx` | modify | Move `'Control'` to position 0 in `CATEGORY_ORDER` |
| `frontend/src/components/Canvas/QuickNodeSearch.tsx` | modify | Boost the Start node to top of results when query is empty or matches "start" |
| `frontend/src/utils/autoLayout.ts` | create | `pickTargets`, `findConnectedComponents`, `layoutComponentWithDagre`, `packIntoSwimLanes`, top-level `autoLayout()` |
| `frontend/src/utils/autoLayout.test.ts` | create | Vitest unit tests for all six layout cases from the spec |
| `frontend/src/utils/findEntryPoints.ts` | create | Client-side mirror of backend `find_entry_points()` (used to short-circuit run with toast) |
| `frontend/src/utils/findEntryPoints.test.ts` | create | Vitest tests matching the backend test cases |
| `frontend/src/store/tabStore.ts` | modify | Add `toggleEntryPoint(nodeId)` and `applyLayout(mode)` actions; layout pushes to existing `undoStack` |
| `frontend/src/store/uiStore.ts` | modify | Add `lastLayoutMode: LayoutMode` with localStorage persistence (key `codefyui-last-layout-mode`) |
| `frontend/src/components/Toolbar/Toolbar.tsx` | modify | Add Auto Layout split button (left button + caret dropdown) in the right cluster |
| `frontend/src/components/Toolbar/Toolbar.module.css` | modify | Add `.splitButton`, `.splitButtonCaret`, `.layoutDropdown` styles |
| `frontend/src/hooks/useKeyboardShortcuts.ts` | modify | Add `Shift+L` binding |
| `frontend/src/hooks/useGraphExecution.ts` | modify | Pre-check entry points before sending run; toast and abort if none |
| `frontend/src/components/shared/MigrationModal.tsx` | create | Modal shown on graph load when no entry points exist |
| `frontend/src/components/shared/MigrationModal.module.css` | create | Migration modal styles |
| `frontend/src/i18n/locales/en.ts` | modify | New keys for toolbar/context-menu/migration/execution/palette |
| `frontend/src/i18n/locales/zh-TW.ts` | modify | Chinese translations of all new keys |
| `scripts/migrate_entry_points.py` | create | One-shot helper that walks `examples/**/graph.json` and adds `isEntryPoint: true` to data-roots |
| `examples/**/graph.json` (×16) | modify | Auto-modified by the migration script |

---

## Phase 1: Backend foundation

### Task 1: Add `DataType.TRIGGER` enum + type_system compatibility

**Files:**
- Modify: `backend/app/core/node_base.py:9-21`
- Modify: `backend/app/core/type_system.py:1-23`
- Test: `backend/tests/test_type_system.py` (create if missing, otherwise add to existing)

**Context:** `DataType` is a string enum with 11 values (TENSOR, MODEL, DATASET, etc.). `_COMPAT` maps each source type to a set of compatible targets. `ANY` is the universal type — it currently includes every other DataType. Trigger ports must NOT connect to data ports, so `TRIGGER` must be excluded from `ANY`'s compatible set.

- [ ] **Step 1: Read current state**

```bash
# Read both files end-to-end
```

Files: `backend/app/core/node_base.py` and `backend/app/core/type_system.py`. Confirm the current `DataType` enum has 11 values and `_COMPAT` uses a dict-comprehension `{dt for dt in DataType}` for ANY.

- [ ] **Step 2: Write the failing test**

Create or append to `backend/tests/test_type_system.py`:

```python
from app.core.node_base import DataType
from app.core.type_system import is_compatible


def test_trigger_only_connects_to_trigger():
    assert is_compatible(DataType.TRIGGER, DataType.TRIGGER) is True


def test_trigger_does_not_connect_to_data():
    assert is_compatible(DataType.TRIGGER, DataType.TENSOR) is False
    assert is_compatible(DataType.TRIGGER, DataType.DATASET) is False
    assert is_compatible(DataType.TRIGGER, DataType.MODEL) is False


def test_data_does_not_connect_to_trigger():
    assert is_compatible(DataType.TENSOR, DataType.TRIGGER) is False
    assert is_compatible(DataType.DATASET, DataType.TRIGGER) is False


def test_any_does_not_swallow_trigger():
    """ANY is universal for data types but must NOT match TRIGGER, otherwise
    a trigger output could be connected to any data input."""
    assert is_compatible(DataType.ANY, DataType.TRIGGER) is False
    assert is_compatible(DataType.TRIGGER, DataType.ANY) is False
```

- [ ] **Step 3: Run tests, watch them fail**

Run from `backend/`:

```bash
uv run pytest tests/test_type_system.py -v
```

Expected: tests fail with `AttributeError: TRIGGER` (because the enum value doesn't exist yet).

- [ ] **Step 4: Add `TRIGGER` to the enum**

Edit `backend/app/core/node_base.py` — find the `DataType` class (line 9) and add `TRIGGER = "TRIGGER"` after `ANY = "ANY"`:

```python
class DataType(str, Enum):
    TENSOR = "TENSOR"
    MODEL = "MODEL"
    DATASET = "DATASET"
    DATALOADER = "DATALOADER"
    OPTIMIZER = "OPTIMIZER"
    LOSS_FN = "LOSS_FN"
    SCALAR = "SCALAR"
    STRING = "STRING"
    IMAGE = "IMAGE"
    LIST = "LIST"
    ANY = "ANY"
    TRIGGER = "TRIGGER"
```

- [ ] **Step 5: Update `_COMPAT` to exclude TRIGGER from ANY's set**

Edit `backend/app/core/type_system.py`. Replace the entire `_COMPAT` dict with:

```python
_COMPAT: dict[DataType, set[DataType]] = {
    DataType.TENSOR: {DataType.TENSOR, DataType.ANY},
    DataType.MODEL: {DataType.MODEL, DataType.ANY},
    DataType.DATASET: {DataType.DATASET, DataType.ANY},
    DataType.DATALOADER: {DataType.DATALOADER, DataType.ANY},
    DataType.OPTIMIZER: {DataType.OPTIMIZER, DataType.ANY},
    DataType.LOSS_FN: {DataType.LOSS_FN, DataType.ANY},
    DataType.SCALAR: {DataType.SCALAR, DataType.ANY},
    DataType.STRING: {DataType.STRING, DataType.ANY},
    DataType.IMAGE: {DataType.IMAGE, DataType.TENSOR, DataType.ANY},
    DataType.LIST: {DataType.LIST, DataType.ANY},
    DataType.ANY: {dt for dt in DataType if dt is not DataType.TRIGGER},
    DataType.TRIGGER: {DataType.TRIGGER},
}
```

Then update `is_compatible()` so that `target == DataType.ANY` no longer auto-passes when source is TRIGGER:

```python
def is_compatible(source: DataType, target: DataType) -> bool:
    if source == DataType.TRIGGER or target == DataType.TRIGGER:
        return source == DataType.TRIGGER and target == DataType.TRIGGER
    if target == DataType.ANY:
        return True
    return target in _COMPAT.get(source, set())
```

- [ ] **Step 6: Run tests, watch them pass**

```bash
uv run pytest tests/test_type_system.py -v
```

Expected: all 4 new tests pass.

- [ ] **Step 7: Run the full backend test suite**

```bash
uv run pytest -x
```

Expected: no regressions. Existing tests for other DataTypes still pass.

- [ ] **Step 8: Commit**

```bash
git add backend/app/core/node_base.py backend/app/core/type_system.py backend/tests/test_type_system.py
git commit -m "feat(backend): add DataType.TRIGGER for execution markers"
```

---

### Task 2: Add `isEntryPoint` and edge `type` schema fields

**Files:**
- Modify: `backend/app/schemas/models.py:34-51`
- Test: `backend/tests/test_schemas.py` (create if missing)

**Context:** `NodeData` is a Pydantic model with `id`, `type`, `position`, and `data` fields. `EdgeData` has `id`, `source`, `target`, `sourceHandle`, `targetHandle`. `data` on `NodeData` is currently a free-form `dict[str, Any]` — the entry-point flag belongs on `NodeData` itself, not nested inside `data`, so it shows up as a top-level field on the wire.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_schemas.py`:

```python
from app.schemas.models import NodeData, EdgeData


def test_node_data_default_is_entry_point_false():
    n = NodeData(id="n1", type="Dataset")
    assert n.isEntryPoint is False


def test_node_data_can_set_is_entry_point():
    n = NodeData(id="n1", type="Dataset", isEntryPoint=True)
    assert n.isEntryPoint is True


def test_edge_data_default_type_is_data():
    e = EdgeData(id="e1", source="a", target="b")
    assert e.type == "data"


def test_edge_data_can_set_type_trigger():
    e = EdgeData(id="e1", source="a", target="b", type="trigger")
    assert e.type == "trigger"


def test_edge_data_rejects_unknown_type():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EdgeData(id="e1", source="a", target="b", type="bogus")
```

- [ ] **Step 2: Run tests, watch them fail**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: failures because `isEntryPoint` and `type` don't exist yet on the models.

- [ ] **Step 3: Add fields to `NodeData` and `EdgeData`**

Edit `backend/app/schemas/models.py`. Find the `NodeData` class (around line 34) and add `isEntryPoint` field. Find `EdgeData` and add `type` field. Also import `Literal`:

```python
from typing import Any, Literal


class NodeData(BaseModel):
    id: str
    type: str
    position: dict[str, float] = {"x": 0, "y": 0}
    data: dict[str, Any] = {}
    isEntryPoint: bool = False  # NEW


class EdgeData(BaseModel):
    id: str
    source: str
    target: str
    sourceHandle: str = ""
    targetHandle: str = ""
    type: Literal["data", "trigger"] = "data"  # NEW
```

- [ ] **Step 4: Run tests, watch them pass**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Run full backend suite**

```bash
uv run pytest -x
```

Expected: no regressions. Old graph.json files (which lack these fields) still parse via the defaults.

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/models.py backend/tests/test_schemas.py
git commit -m "feat(backend): add isEntryPoint and edge type fields to schemas"
```

---

### Task 3: Create `StartNode`

**Files:**
- Create: `backend/app/nodes/control/__init__.py`
- Create: `backend/app/nodes/control/start_node.py`
- Test: `backend/tests/test_start_node.py` (new)

**Context:** `BaseNode` uses classmethods `define_inputs() / define_outputs() / define_params()` (NOT class-level constants). `NODE_NAME` is the registered identifier. Node auto-discovery walks `backend/app/nodes/**` so dropping a new file in a new subdirectory works without registry edits — see `backend/app/core/node_registry.py` and `backend/app/nodes/data/dataset_node.py:6-76` for the canonical pattern.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_start_node.py`:

```python
from app.core.node_base import DataType
from app.nodes.control.start_node import StartNode


def test_start_node_metadata():
    assert StartNode.NODE_NAME == "Start"
    assert StartNode.CATEGORY == "Control"
    assert StartNode.DESCRIPTION  # non-empty


def test_start_node_has_no_inputs():
    assert StartNode.define_inputs() == []


def test_start_node_has_one_trigger_output():
    outputs = StartNode.define_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "trigger"
    assert outputs[0].data_type == DataType.TRIGGER


def test_start_node_has_no_params():
    assert StartNode.define_params() == []


def test_start_node_execute_is_noop():
    node = StartNode()
    result = node.execute(inputs={}, params={})
    assert result == {}


def test_start_node_is_auto_discovered():
    """The node registry should pick up StartNode after discovery."""
    from app.core.node_registry import NodeRegistry
    reg = NodeRegistry()
    reg.discover("app/nodes", "app.nodes")
    assert reg.get("Start") is StartNode
```

- [ ] **Step 2: Run tests, watch them fail**

```bash
uv run pytest tests/test_start_node.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.nodes.control'`.

- [ ] **Step 3: Create the package marker**

Create empty file `backend/app/nodes/control/__init__.py`.

- [ ] **Step 4: Implement `StartNode`**

Create `backend/app/nodes/control/start_node.py`:

```python
"""Start node — marks an execution entry point.

A `StartNode` has no inputs and one `trigger` output. It does no work at
runtime; its only purpose is to declare that the connected component
containing it is "live" and should be executed. Connect the trigger output
to a data-root node (e.g. Dataset) to mark that node as an entry point.
"""

from typing import Any

from app.core.node_base import BaseNode, DataType, PortDefinition


class StartNode(BaseNode):
    NODE_NAME = "Start"
    CATEGORY = "Control"
    DESCRIPTION = (
        "Marks an execution entry point. Connect this to the first node "
        "of the script you want to run, like a 'When Flag Clicked' block."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="trigger",
                data_type=DataType.TRIGGER,
                description="Execution trigger marker (carries no data)",
            ),
        ]

    @classmethod
    def define_params(cls) -> list:
        return []

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        return {}
```

- [ ] **Step 5: Run tests, watch them pass**

```bash
uv run pytest tests/test_start_node.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 6: Run full suite**

```bash
uv run pytest -x
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/nodes/control/__init__.py backend/app/nodes/control/start_node.py backend/tests/test_start_node.py
git commit -m "feat(backend): add StartNode for marking execution entry points"
```

---

### Task 4: Add `find_entry_points` and `reachable_from_entry_points` helpers

**Files:**
- Modify: `backend/app/core/graph_engine.py`
- Test: `backend/tests/test_graph_engine.py` (extend)

**Context:** These two pure functions are the foundation for entry-point gating. `find_entry_points` returns the ids of nodes that are entry points (either `isEntryPoint=True` or have an incoming `trigger` edge). `reachable_from_entry_points` does BFS forward through DATA edges only (trigger edges are markers, not data dependencies). Both helpers operate on dict-shaped nodes/edges (the engine uses `model_dump()` everywhere — see `backend/app/api/routes_graph.py:14-19`).

- [ ] **Step 1: Read existing graph_engine.py shape**

Read `backend/app/core/graph_engine.py` lines 1-50 to confirm imports and helper-function placement style. The new helpers go near `_has_cycle()` (around line 218).

- [ ] **Step 2: Write the failing tests**

Append to `backend/tests/test_graph_engine.py`:

```python
from app.core.graph_engine import find_entry_points, reachable_from_entry_points


def test_find_entry_points_explicit_marker():
    nodes = [
        {"id": "a", "isEntryPoint": True},
        {"id": "b", "isEntryPoint": False},
        {"id": "c", "isEntryPoint": False},
    ]
    edges = []
    assert find_entry_points(nodes, edges) == ["a"]


def test_find_entry_points_via_trigger_edge():
    nodes = [
        {"id": "start", "type": "Start", "isEntryPoint": False},
        {"id": "ds", "type": "Dataset", "isEntryPoint": False},
    ]
    edges = [
        {"id": "e1", "source": "start", "target": "ds", "type": "trigger"},
    ]
    # The DOWNSTREAM node (ds) is the entry point because it has an
    # incoming trigger edge.
    assert "ds" in find_entry_points(nodes, edges)
    # Start node itself is also an entry (it has isEntryPoint by virtue
    # of being a Start? — no: Start nodes are entry points because they're
    # always treated as such; we'll handle that via isEntryPoint=True being
    # set when the StartNode is instantiated on the canvas, OR by treating
    # Start type as implicit entry. We'll go with implicit-by-type below.)
    assert "start" in find_entry_points(nodes, edges)


def test_find_entry_points_combined():
    nodes = [
        {"id": "a", "type": "Dataset", "isEntryPoint": True},
        {"id": "start", "type": "Start", "isEntryPoint": False},
        {"id": "b", "type": "Dataset", "isEntryPoint": False},
        {"id": "c", "type": "Conv", "isEntryPoint": False},
    ]
    edges = [
        {"id": "e1", "source": "start", "target": "b", "type": "trigger"},
        {"id": "e2", "source": "a", "target": "c", "type": "data"},
    ]
    result = set(find_entry_points(nodes, edges))
    assert result == {"a", "b", "start"}


def test_find_entry_points_none():
    nodes = [{"id": "a"}, {"id": "b"}]
    edges = [{"id": "e1", "source": "a", "target": "b", "type": "data"}]
    assert find_entry_points(nodes, edges) == []


def test_reachable_traverses_data_edges_only():
    nodes = [{"id": n} for n in ["start", "ds", "dl", "model"]]
    edges = [
        {"id": "e1", "source": "start", "target": "ds", "type": "trigger"},
        {"id": "e2", "source": "ds", "target": "dl", "type": "data"},
        {"id": "e3", "source": "dl", "target": "model", "type": "data"},
    ]
    # Starting from `ds` (data root that received a trigger), BFS through
    # data edges should reach ds, dl, model — but NOT start (it's upstream
    # via a trigger edge, which is not traversed).
    reachable = reachable_from_entry_points(["ds"], edges)
    assert reachable == {"ds", "dl", "model"}


def test_reachable_includes_start_when_explicitly_in_seed():
    nodes = [{"id": n} for n in ["start", "ds"]]
    edges = [{"id": "e1", "source": "start", "target": "ds", "type": "trigger"}]
    # If we seed BFS with start AND ds, both are in the result; trigger
    # edges are still not traversed, but the seeds themselves are included.
    reachable = reachable_from_entry_points(["start", "ds"], edges)
    assert reachable == {"start", "ds"}


def test_reachable_handles_disconnected_components():
    nodes = [{"id": n} for n in ["a", "b", "x", "y"]]
    edges = [
        {"id": "e1", "source": "a", "target": "b", "type": "data"},
        {"id": "e2", "source": "x", "target": "y", "type": "data"},
    ]
    assert reachable_from_entry_points(["a"], edges) == {"a", "b"}
```

**Note on Start-node-as-entry semantic:** A `StartNode` should be treated as an entry point implicitly. The simplest way to express this is: in `find_entry_points`, also include any node whose `type == "Start"`. This avoids requiring the frontend to set `isEntryPoint=True` on Start nodes (the user shouldn't have to mark them — they're always entries).

- [ ] **Step 3: Run tests, watch them fail**

```bash
uv run pytest tests/test_graph_engine.py::test_find_entry_points_explicit_marker -v
```

Expected: `ImportError` because `find_entry_points` doesn't exist yet.

- [ ] **Step 4: Implement the helpers**

In `backend/app/core/graph_engine.py`, add these two functions after `_has_cycle()` (around line 240):

```python
def find_entry_points(
    nodes: list[dict],
    edges: list[dict],
) -> list[str]:
    """Return ids of nodes that are entry points.

    A node is an entry point if any of:
      1. Its `isEntryPoint` field is True.
      2. It is of type "Start" (Start nodes are always entry points).
      3. It has at least one incoming edge of type "trigger".

    The order of returned ids matches the order in `nodes` for determinism.
    """
    entry_ids: list[str] = []
    nodes_with_trigger_in: set[str] = {
        e["target"]
        for e in edges
        if e.get("type", "data") == "trigger"
    }
    for node in nodes:
        nid = node["id"]
        is_marker = bool(node.get("isEntryPoint", False))
        is_start_type = node.get("type") == "Start"
        has_trigger_in = nid in nodes_with_trigger_in
        if is_marker or is_start_type or has_trigger_in:
            entry_ids.append(nid)
    return entry_ids


def reachable_from_entry_points(
    entry_ids: list[str],
    edges: list[dict],
) -> set[str]:
    """BFS forward from entry_ids through DATA edges only.

    Trigger edges are markers, not data dependencies, and are not
    traversed. The seed entry_ids themselves are always included in the
    result, regardless of edge types.
    """
    reachable: set[str] = set(entry_ids)
    frontier: list[str] = list(entry_ids)
    # Build adjacency list of data edges only.
    adj: dict[str, list[str]] = {}
    for e in edges:
        if e.get("type", "data") == "data":
            adj.setdefault(e["source"], []).append(e["target"])
    while frontier:
        node = frontier.pop()
        for next_node in adj.get(node, []):
            if next_node not in reachable:
                reachable.add(next_node)
                frontier.append(next_node)
    return reachable
```

- [ ] **Step 5: Run the new tests**

```bash
uv run pytest tests/test_graph_engine.py -v -k "entry_points or reachable"
```

Expected: all 7 new tests pass.

- [ ] **Step 6: Run the full graph_engine test file**

```bash
uv run pytest tests/test_graph_engine.py -v
```

Expected: existing tests still pass (helpers are pure additions).

- [ ] **Step 7: Commit**

```bash
git add backend/app/core/graph_engine.py backend/tests/test_graph_engine.py
git commit -m "feat(backend): add find_entry_points and reachable_from_entry_points helpers"
```

---

### Task 5: Update `validate_graph()` to enforce entry-point rules

**Files:**
- Modify: `backend/app/core/graph_engine.py:110-215`
- Test: `backend/tests/test_graph_engine.py` (extend)

**Context:** `validate_graph()` currently returns `list[str]` of error messages. We add 2 new validation rules: (1) graph must have ≥1 entry point; (2) entry-point nodes must have no incoming **data** edges (entry points are data-roots — Scratch hat-block semantic). Cycle detection should also be scoped to the *executable subgraph* so a draft component with a temporary cycle doesn't fail validation.

- [ ] **Step 1: Read current `validate_graph` body**

Read `backend/app/core/graph_engine.py:110-215` to confirm the structure of error collection.

- [ ] **Step 2: Write the failing tests**

Append to `backend/tests/test_graph_engine.py`:

```python
from app.core.graph_engine import validate_graph


def _make_node(nid, ntype="Dataset", is_entry=False):
    return {
        "id": nid,
        "type": ntype,
        "isEntryPoint": is_entry,
        "data": {"params": {}},
    }


def _make_edge(eid, src, tgt, etype="data"):
    return {
        "id": eid,
        "source": src,
        "target": tgt,
        "sourceHandle": "out",
        "targetHandle": "in",
        "type": etype,
    }


def test_validate_rejects_no_entry_points():
    nodes = [_make_node("a"), _make_node("b")]
    edges = [_make_edge("e1", "a", "b")]
    errors = validate_graph(nodes, edges)
    assert any("entry point" in err.lower() for err in errors)


def test_validate_accepts_single_entry_point():
    nodes = [_make_node("a", is_entry=True), _make_node("b")]
    edges = [_make_edge("e1", "a", "b")]
    errors = validate_graph(nodes, edges)
    # Should pass entry-point check (other validation may still complain
    # about node type registration; we only care entry-point rule passes)
    assert not any("entry point" in err.lower() for err in errors)


def test_validate_rejects_entry_with_incoming_data_edge():
    """A node marked isEntryPoint=True must not have incoming data edges."""
    nodes = [_make_node("a"), _make_node("b", is_entry=True)]
    edges = [_make_edge("e1", "a", "b", etype="data")]
    errors = validate_graph(nodes, edges)
    assert any("data-root" in err.lower() or "incoming" in err.lower() for err in errors)


def test_validate_rejects_trigger_target_with_incoming_data_edge():
    """A node that is an entry via incoming trigger ALSO must be a data-root."""
    nodes = [
        _make_node("start", ntype="Start"),
        _make_node("upstream"),
        _make_node("target"),
    ]
    edges = [
        _make_edge("e1", "start", "target", etype="trigger"),
        _make_edge("e2", "upstream", "target", etype="data"),
    ]
    errors = validate_graph(nodes, edges)
    assert any("data-root" in err.lower() or "incoming" in err.lower() for err in errors)


def test_validate_allows_cycle_in_draft_component():
    """A cycle inside a non-entry-pointed (draft) component should NOT
    fail validation, because the draft is skipped at execution."""
    nodes = [
        _make_node("ep", is_entry=True),
        _make_node("a"),
        _make_node("b"),
        _make_node("c"),
    ]
    edges = [
        # Cycle in the draft component a→b→c→a
        _make_edge("e1", "a", "b"),
        _make_edge("e2", "b", "c"),
        _make_edge("e3", "c", "a"),
    ]
    errors = validate_graph(nodes, edges)
    assert not any("cycle" in err.lower() for err in errors)


def test_validate_rejects_cycle_in_entry_pointed_component():
    """A cycle in an entry-pointed component fails (current behaviour)."""
    nodes = [
        _make_node("ep", is_entry=True),
        _make_node("a"),
        _make_node("b"),
    ]
    edges = [
        _make_edge("e1", "ep", "a"),
        _make_edge("e2", "a", "b"),
        _make_edge("e3", "b", "ep"),  # cycle back
    ]
    errors = validate_graph(nodes, edges)
    # Note: this will also fail rule "entry must be data-root" because ep
    # has an incoming edge from b. So we accept either error here.
    assert any(("cycle" in err.lower()) or ("data-root" in err.lower()) for err in errors)
```

- [ ] **Step 3: Run, watch fail**

```bash
uv run pytest tests/test_graph_engine.py -v -k "validate"
```

Expected: failures for the new tests.

- [ ] **Step 4: Update `validate_graph()`**

In `backend/app/core/graph_engine.py`, locate `validate_graph()` (around line 110). Add the new validation rules at the end of the function, BEFORE the existing `_has_cycle()` call. Replace the call to `_has_cycle()` with a scoped version. Here is the structure to add (preserve the existing rules above):

```python
def validate_graph(
    nodes: list[dict],
    edges: list[dict],
) -> list[str]:
    errors: list[str] = []

    # ... existing structural validation (keep as-is) ...

    # NEW: Entry-point rules
    entry_ids = find_entry_points(nodes, edges)
    if not entry_ids:
        errors.append(
            "Graph has no entry points. Mark a root node as an entry "
            "point or add a Start node."
        )
        # Still run remaining checks so user sees all problems at once
        executable_node_ids = {n["id"] for n in nodes}  # treat all as exec for further checks
    else:
        executable_node_ids = reachable_from_entry_points(entry_ids, edges)

    # NEW: Entry-point nodes must have no incoming DATA edges
    for entry_id in entry_ids:
        incoming_data = [
            e for e in edges
            if e["target"] == entry_id and e.get("type", "data") == "data"
        ]
        if incoming_data:
            errors.append(
                f"Node '{entry_id}' is an entry point but has incoming "
                f"data edges. Entry points must be data-roots."
            )

    # MODIFIED: Run cycle detection on the EXECUTABLE subgraph only.
    # Drafts (nodes outside executable_node_ids) are skipped.
    executable_nodes = [n for n in nodes if n["id"] in executable_node_ids]
    executable_edges = [
        e for e in edges
        if e["source"] in executable_node_ids
        and e["target"] in executable_node_ids
        and e.get("type", "data") == "data"
    ]
    if _has_cycle(executable_nodes, executable_edges):
        errors.append("Graph contains a cycle")

    return errors
```

**Important:** The existing `_has_cycle()` call on the full graph must be REMOVED — the new scoped version replaces it. Search for the existing call site and delete it.

- [ ] **Step 5: Run new tests**

```bash
uv run pytest tests/test_graph_engine.py -v -k "validate"
```

Expected: all new tests pass.

- [ ] **Step 6: Run full suite, fix any regressions**

```bash
uv run pytest -x
```

If existing tests fail because they don't have entry points, **do NOT auto-fix the test fixtures yet** — those will be migrated in Task 6 alongside the execute_graph changes.

If a test fails simply because the helper graph it constructs has no entry point, mark it XFAIL temporarily with a note `# Updated in Task 6` and proceed.

- [ ] **Step 7: Commit**

```bash
git add backend/app/core/graph_engine.py backend/tests/test_graph_engine.py
git commit -m "feat(backend): enforce entry-point rules in validate_graph"
```

---

### Task 6: Update `topological_levels` and `execute_graph` for reachability + trigger exclusion

**Files:**
- Modify: `backend/app/core/graph_engine.py:266-534`
- Modify: `backend/tests/test_graph_engine.py` (fix any tests left XFAIL in Task 5; add new ones)

**Context:** Two changes: (1) when computing in-degree, edges of type `"trigger"` are EXCLUDED so a node receiving a trigger from a Start is still treated as a root; (2) `execute_graph` filters its node set to `reachable_from_entry_points()` so drafts are silently skipped. The Start node itself is in the executable set (it's an entry) but its `execute()` is a no-op so it adds nothing to outputs.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_graph_engine.py`:

```python
import pytest
from app.core.graph_engine import topological_levels, execute_graph


def test_topological_levels_excludes_trigger_from_in_degree():
    """A Dataset receiving a trigger from a Start should still be at level 0."""
    nodes = [
        {"id": "start", "type": "Start", "isEntryPoint": False, "data": {"params": {}}},
        {"id": "ds", "type": "Dataset", "isEntryPoint": False, "data": {"params": {}}},
        {"id": "dl", "type": "DataLoader", "isEntryPoint": False, "data": {"params": {}}},
    ]
    edges = [
        {"id": "e1", "source": "start", "target": "ds", "sourceHandle": "trigger", "targetHandle": "trigger", "type": "trigger"},
        {"id": "e2", "source": "ds", "target": "dl", "sourceHandle": "dataset", "targetHandle": "dataset", "type": "data"},
    ]
    levels = topological_levels(nodes, edges)
    # Both start and ds should be in level 0 (start has no inputs, ds's
    # only incoming edge is a trigger which is excluded).
    assert "start" in levels[0]
    assert "ds" in levels[0]
    assert "dl" in levels[1]


@pytest.mark.asyncio
async def test_execute_graph_skips_draft_components():
    """Draft components (no entry point) should be skipped silently."""
    # We need real registered nodes for execute_graph to actually run.
    # Use Print node as a stand-in (assumed to exist; otherwise use the
    # _TestSource pattern from existing tests).
    nodes = [
        {"id": "live", "type": "_TestSource", "isEntryPoint": True, "data": {"params": {"value": 42}}},
        {"id": "draft", "type": "_TestSource", "isEntryPoint": False, "data": {"params": {"value": 99}}},
    ]
    edges = []
    results = await execute_graph(nodes, edges)
    assert "live" in results
    assert "draft" not in results
```

- [ ] **Step 2: Run, watch fail**

```bash
uv run pytest tests/test_graph_engine.py -v -k "trigger or skips_draft"
```

Expected: failures.

- [ ] **Step 3: Update `topological_levels()`**

In `backend/app/core/graph_engine.py:266-294`, find the in-degree calculation. Modify it to exclude trigger edges:

```python
def topological_levels(nodes: list[dict], edges: list[dict]) -> list[list[str]]:
    """Group nodes by topological depth for parallel execution.

    Trigger edges (type="trigger") are excluded from in-degree calculation
    because they are markers, not data dependencies.
    """
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    adj: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for edge in edges:
        if edge.get("type", "data") == "trigger":
            continue  # markers, not dependencies
        if edge["target"] in in_degree:
            in_degree[edge["target"]] += 1
        if edge["source"] in adj:
            adj[edge["source"]].append(edge["target"])

    # ... existing level-grouping loop using in_degree and adj
```

Apply the same `if edge.get("type", "data") == "trigger": continue` filter to `topological_sort()` (line 240) and `_has_cycle()` (line 218) for consistency.

- [ ] **Step 4: Update `execute_graph()` to filter to reachable nodes**

In `backend/app/core/graph_engine.py:297-534`, near the start of `execute_graph()` (after preset expansion, before `topological_levels()` is called), insert the reachability filter:

```python
async def execute_graph(
    nodes: list[dict],
    edges: list[dict],
    on_progress=None,
    context=None,
    error_mode: str = "fail_fast",
    max_retries: int = 0,
    cache=None,
    changed_nodes: list[str] | None = None,
) -> dict[str, Any]:
    # ... existing preset expansion ...

    # NEW: filter to executable subgraph (nodes reachable from entry points)
    entry_ids = find_entry_points(nodes, edges)
    if not entry_ids:
        # validate_graph would catch this, but defend in depth
        raise GraphValidationError("Graph has no entry points")

    executable_ids = reachable_from_entry_points(entry_ids, edges)
    nodes = [n for n in nodes if n["id"] in executable_ids]
    edges = [
        e for e in edges
        if e["source"] in executable_ids and e["target"] in executable_ids
    ]

    # ... rest of execute_graph (validate_graph, topological_levels, level loop) ...
```

- [ ] **Step 5: Fix any XFAIL'd tests from Task 5**

Find any tests marked XFAIL in Task 5 and update their fixtures to include `isEntryPoint=True` on the appropriate root node. Remove the XFAIL marker.

- [ ] **Step 6: Run new tests**

```bash
uv run pytest tests/test_graph_engine.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Run full backend suite**

```bash
uv run pytest -x
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add backend/app/core/graph_engine.py backend/tests/test_graph_engine.py
git commit -m "feat(backend): filter execution to entry-point reachable subgraph"
```

---

## Phase 2: Frontend foundation

### Task 7: Add `@dagrejs/dagre` dependency

**Files:**
- Modify: `frontend/package.json:13-17`

- [ ] **Step 1: Read current dependencies block**

Read `frontend/package.json` to confirm the structure.

- [ ] **Step 2: Install via pnpm**

Run from `frontend/`:

```bash
pnpm add @dagrejs/dagre
```

Expected: pnpm resolves and installs `@dagrejs/dagre` (~30KB), updates `package.json` and `pnpm-lock.yaml`.

- [ ] **Step 3: Verify the import works**

Run from `frontend/`:

```bash
node -e "import('@dagrejs/dagre').then(d => console.log('OK', typeof d.default.graphlib.Graph))"
```

Expected: `OK function`.

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/pnpm-lock.yaml
git commit -m "chore(frontend): add @dagrejs/dagre for auto layout"
```

---

### Task 8: Entry-point visual on `BaseNode`

**Files:**
- Create: `frontend/src/components/Nodes/flag-icon.svg`
- Modify: `frontend/src/components/Nodes/BaseNode.module.css`
- Modify: `frontend/src/components/Nodes/BaseNode.tsx`
- Modify: `frontend/src/types/index.ts` (or wherever `NodeData` is typed)

**Context:** `BaseNode.tsx` already takes `data` and applies CSS modifiers via `className`. We add an `.entryPoint` modifier class that shows a green outline + a flag badge in the top-left corner. The flag SVG is a single-color silhouette using `currentColor` so it inherits the outline color.

- [ ] **Step 1: Discover where `NodeData` is typed**

Run:

```bash
# Use Grep to find NodeData type definition
```

Grep `frontend/src/types` for `interface NodeData` and `type NodeData`. The result file is where we add `isEntryPoint?: boolean`.

- [ ] **Step 2: Create the flag icon**

Create `frontend/src/components/Nodes/flag-icon.svg`:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none">
  <path d="M3 1 V15 M3 2 H12 L10 5 L12 8 H3" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" fill="currentColor" fill-opacity="0.4"/>
</svg>
```

- [ ] **Step 3: Add `.entryPoint` modifier to `BaseNode.module.css`**

Open `frontend/src/components/Nodes/BaseNode.module.css`. At the end of the file, append:

```css
/* Entry point marker — applied when data.isEntryPoint === true */
.entryPoint {
  outline: 2px solid #22c55e;
  outline-offset: 3px;
  box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.18), 0 4px 12px rgba(0, 0, 0, 0.4);
}

.entryPoint::before {
  content: "";
  position: absolute;
  top: -10px;
  left: -10px;
  width: 18px;
  height: 18px;
  background-color: #22c55e;
  border-radius: 50%;
  border: 2px solid #0a0a0a;
  background-image: url("./flag-icon.svg");
  background-size: 12px 12px;
  background-repeat: no-repeat;
  background-position: center;
  color: #ffffff;
  z-index: 10;
}
```

- [ ] **Step 4: Add `isEntryPoint` to `NodeData` type**

Open the file from Step 1. Add the optional field:

```typescript
export interface NodeData {
  // ... existing fields
  isEntryPoint?: boolean;
}
```

- [ ] **Step 5: Apply `.entryPoint` modifier in `BaseNode.tsx`**

Open `frontend/src/components/Nodes/BaseNode.tsx`. Find the `className={styles.node}` (around line 50-60). Change it to conditionally include `.entryPoint`:

```tsx
className={`${styles.node} ${data.isEntryPoint ? styles.entryPoint : ''}`}
```

- [ ] **Step 6: Visual smoke test (manual, since no UI test framework for CSS)**

Run from `frontend/`:

```bash
pnpm dev
```

Open the app. Manually edit `localStorage` to set `isEntryPoint: true` on a node and reload. Expect the green outline + flag badge.

(This is a manual check — no Vitest test for CSS rendering. Chrome MCP verification in Task 23 will exercise the full flow.)

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/Nodes/flag-icon.svg frontend/src/components/Nodes/BaseNode.module.css frontend/src/components/Nodes/BaseNode.tsx frontend/src/types/index.ts
git commit -m "feat(frontend): add entry-point visual to BaseNode"
```

---

### Task 9: Generic `ContextMenu` primitive

**Files:**
- Create: `frontend/src/components/shared/ContextMenu.tsx`
- Create: `frontend/src/components/shared/ContextMenu.module.css`
- Test: `frontend/src/components/shared/ContextMenu.test.tsx`

**Context:** No reusable context-menu primitive exists in the codebase. We create a minimal one: positioned at cursor coordinates, single-level items, click-outside dismissal. It will be reused beyond this feature, so keep the API generic.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/shared/ContextMenu.test.tsx`:

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ContextMenu } from './ContextMenu';

describe('ContextMenu', () => {
  it('renders items at the given position', () => {
    const onSelect = vi.fn();
    render(
      <ContextMenu
        x={100}
        y={200}
        items={[{ id: 'foo', label: 'Foo Action' }]}
        onSelect={onSelect}
        onClose={() => {}}
      />
    );
    const item = screen.getByText('Foo Action');
    expect(item).toBeInTheDocument();
  });

  it('fires onSelect with item id when clicked', () => {
    const onSelect = vi.fn();
    const onClose = vi.fn();
    render(
      <ContextMenu
        x={0}
        y={0}
        items={[{ id: 'foo', label: 'Foo' }, { id: 'bar', label: 'Bar' }]}
        onSelect={onSelect}
        onClose={onClose}
      />
    );
    fireEvent.click(screen.getByText('Bar'));
    expect(onSelect).toHaveBeenCalledWith('bar');
    expect(onClose).toHaveBeenCalled();
  });

  it('closes on outside click', () => {
    const onClose = vi.fn();
    render(
      <ContextMenu
        x={0}
        y={0}
        items={[{ id: 'foo', label: 'Foo' }]}
        onSelect={() => {}}
        onClose={onClose}
      />
    );
    fireEvent.mouseDown(document.body);
    expect(onClose).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Check if `@testing-library/react` is installed**

```bash
# from frontend/
cat package.json | grep testing-library
```

If not installed, run:

```bash
pnpm add -D @testing-library/react @testing-library/jest-dom jsdom
```

Make sure `vitest.config.ts` has `environment: 'jsdom'`. If not, add it.

- [ ] **Step 3: Run, watch fail**

```bash
pnpm vitest run src/components/shared/ContextMenu.test.tsx
```

Expected: failure (`ContextMenu` does not exist yet).

- [ ] **Step 4: Implement `ContextMenu`**

Create `frontend/src/components/shared/ContextMenu.module.css`:

```css
.menu {
  position: fixed;
  background: #1f1f1f;
  border: 1px solid #444;
  border-radius: 4px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
  padding: 4px 0;
  z-index: 1000;
  min-width: 160px;
}

.item {
  padding: 8px 16px;
  cursor: pointer;
  color: #e0e0e0;
  font-size: 13px;
  user-select: none;
}

.item:hover {
  background: #2d4a2d;
  color: #ffffff;
}
```

Create `frontend/src/components/shared/ContextMenu.tsx`:

```tsx
import { useEffect, useRef } from 'react';
import styles from './ContextMenu.module.css';

export interface ContextMenuItem {
  id: string;
  label: string;
}

export interface ContextMenuProps {
  x: number;
  y: number;
  items: ContextMenuItem[];
  onSelect: (itemId: string) => void;
  onClose: () => void;
}

export function ContextMenu({ x, y, items, onSelect, onClose }: ContextMenuProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  return (
    <div ref={ref} className={styles.menu} style={{ left: x, top: y }}>
      {items.map((item) => (
        <div
          key={item.id}
          className={styles.item}
          onClick={() => {
            onSelect(item.id);
            onClose();
          }}
        >
          {item.label}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 5: Run tests, watch them pass**

```bash
pnpm vitest run src/components/shared/ContextMenu.test.tsx
```

Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/shared/ContextMenu.tsx frontend/src/components/shared/ContextMenu.module.css frontend/src/components/shared/ContextMenu.test.tsx frontend/package.json frontend/pnpm-lock.yaml
git commit -m "feat(frontend): add generic ContextMenu primitive"
```

---

### Task 10: `toggleEntryPoint` action + right-click integration

**Files:**
- Modify: `frontend/src/store/tabStore.ts`
- Modify: `frontend/src/components/Nodes/BaseNode.tsx`
- Test: `frontend/src/store/tabStore.test.ts` (create or extend)

**Context:** Right-click on a node with in-degree 0 (data edges only) should open a one-item context menu: `Set as Entry Point` (or `Remove Entry Point` if already marked). The action toggles `data.isEntryPoint` on the node and pushes the previous state to the existing `undoStack`.

- [ ] **Step 1: Write the failing store test**

Create or append to `frontend/src/store/tabStore.test.ts`:

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { useTabStore } from './tabStore';

describe('toggleEntryPoint', () => {
  beforeEach(() => {
    // Reset store to known state
    useTabStore.setState({ tabs: [], activeTabId: null });
    useTabStore.getState().addTab('test');
  });

  it('sets isEntryPoint to true on a node that was unmarked', () => {
    const tabId = useTabStore.getState().activeTabId!;
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((t) =>
        t.id === tabId
          ? {
              ...t,
              nodes: [
                {
                  id: 'n1',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  data: { id: 'n1', type: 'Dataset', isEntryPoint: false },
                },
              ] as any,
            }
          : t,
      ),
    }));

    useTabStore.getState().toggleEntryPoint('n1');
    const tab = useTabStore.getState().tabs.find((t) => t.id === tabId)!;
    expect(tab.nodes[0].data.isEntryPoint).toBe(true);
  });

  it('clears isEntryPoint on a node that was marked', () => {
    const tabId = useTabStore.getState().activeTabId!;
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((t) =>
        t.id === tabId
          ? {
              ...t,
              nodes: [
                {
                  id: 'n1',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  data: { id: 'n1', type: 'Dataset', isEntryPoint: true },
                },
              ] as any,
            }
          : t,
      ),
    }));

    useTabStore.getState().toggleEntryPoint('n1');
    const tab = useTabStore.getState().tabs.find((t) => t.id === tabId)!;
    expect(tab.nodes[0].data.isEntryPoint).toBe(false);
  });
});
```

- [ ] **Step 2: Run, watch fail**

```bash
pnpm vitest run src/store/tabStore.test.ts
```

Expected: failure (`toggleEntryPoint is not a function`).

- [ ] **Step 3: Add `toggleEntryPoint` to `tabStore`**

Open `frontend/src/store/tabStore.ts`. Find the actions block (after `addTab`). Add this new action:

```typescript
toggleEntryPoint: (nodeId: string) => {
  const tabId = get().activeTabId;
  if (!tabId) return;
  set((state) => ({
    tabs: state.tabs.map((tab) => {
      if (tab.id !== tabId) return tab;
      return {
        ...tab,
        nodes: tab.nodes.map((n) =>
          n.id === nodeId
            ? {
                ...n,
                data: { ...n.data, isEntryPoint: !n.data.isEntryPoint },
              }
            : n,
        ),
      };
    }),
  }));
},
```

Add the type to the store interface:

```typescript
toggleEntryPoint: (nodeId: string) => void;
```

- [ ] **Step 4: Run tests, watch them pass**

```bash
pnpm vitest run src/store/tabStore.test.ts
```

Expected: both tests pass.

- [ ] **Step 5: Wire right-click in `BaseNode.tsx`**

Open `frontend/src/components/Nodes/BaseNode.tsx`. Add state for the context menu and a handler. Imports needed:

```tsx
import { useState } from 'react';
import { ContextMenu, type ContextMenuItem } from '../shared/ContextMenu';
import { useTabStore } from '../../store/tabStore';
import { useReactFlow } from '@xyflow/react';
```

Inside the component, add:

```tsx
const [menuPos, setMenuPos] = useState<{ x: number; y: number } | null>(null);
const toggleEntryPoint = useTabStore((s) => s.toggleEntryPoint);
const { getEdges } = useReactFlow();

const handleContextMenu = (e: React.MouseEvent) => {
  e.preventDefault();
  e.stopPropagation();
  // Only show the menu when this node is a DATA root (no incoming
  // data edges). Trigger edges don't count.
  const edges = getEdges();
  const hasDataIncoming = edges.some(
    (edge) => edge.target === id && (edge.data?.type ?? 'data') === 'data',
  );
  if (hasDataIncoming) return;
  setMenuPos({ x: e.clientX, y: e.clientY });
};

const menuItems: ContextMenuItem[] = [
  {
    id: 'toggle',
    label: data.isEntryPoint ? 'Remove Entry Point' : 'Set as Entry Point',
  },
];
```

Add `onContextMenu={handleContextMenu}` to the root `<div>` and render the menu:

```tsx
{menuPos && (
  <ContextMenu
    x={menuPos.x}
    y={menuPos.y}
    items={menuItems}
    onSelect={(itemId) => {
      if (itemId === 'toggle') toggleEntryPoint(id);
    }}
    onClose={() => setMenuPos(null)}
  />
)}
```

- [ ] **Step 6: Run tests + smoke test**

```bash
pnpm vitest run
pnpm dev
```

In the dev server, manually right-click a Dataset node — context menu should appear. Click "Set as Entry Point" — node gets the green outline.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/store/tabStore.ts frontend/src/store/tabStore.test.ts frontend/src/components/Nodes/BaseNode.tsx
git commit -m "feat(frontend): right-click to mark nodes as entry points"
```

---

### Task 11: `StartNode` component + register in `FlowCanvas` + `triggerEdge` edge type

**Files:**
- Create: `frontend/src/components/Nodes/StartNode.tsx`
- Create: `frontend/src/components/Nodes/StartNode.module.css`
- Create: `frontend/src/components/Canvas/TriggerEdge.tsx`
- Modify: `frontend/src/components/Canvas/FlowCanvas.tsx`

**Context:** `FlowCanvas.tsx` registers node types in a `nodeTypes` object (line 38-41) and edge types in an `edgeTypes` object (or similar). The Start node renders as a small green pill. New edges from a `trigger` source handle should be tagged `type: 'triggerEdge'` so they render dashed green.

- [ ] **Step 1: Create `StartNode.module.css`**

Create `frontend/src/components/Nodes/StartNode.module.css`:

```css
.startNode {
  background: linear-gradient(135deg, #16a34a, #22c55e);
  border-radius: 20px;
  padding: 8px 16px;
  display: flex;
  align-items: center;
  gap: 6px;
  color: #ffffff;
  font-weight: 600;
  font-size: 13px;
  min-width: 80px;
  position: relative;
  box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.18), 0 4px 12px rgba(0, 0, 0, 0.4);
  outline: 2px solid #22c55e;
  outline-offset: 3px;
}

.icon {
  width: 14px;
  height: 14px;
  display: inline-block;
}

.handle {
  background: #22c55e !important;
  width: 12px !important;
  height: 12px !important;
  border: 2px solid #ffffff !important;
  border-radius: 0 !important;
  transform: rotate(45deg) !important;
}
```

- [ ] **Step 2: Implement `StartNode.tsx`**

Create `frontend/src/components/Nodes/StartNode.tsx`:

```tsx
import { Handle, Position, type NodeProps } from '@xyflow/react';
import styles from './StartNode.module.css';

export function StartNode(_: NodeProps) {
  return (
    <div className={styles.startNode}>
      <svg className={styles.icon} viewBox="0 0 16 16" fill="none">
        <path
          d="M3 1 V15 M3 2 H12 L10 5 L12 8 H3"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
          fill="currentColor"
          fillOpacity="0.4"
        />
      </svg>
      <span>Start</span>
      <Handle
        type="source"
        position={Position.Right}
        id="trigger"
        className={styles.handle}
      />
    </div>
  );
}
```

- [ ] **Step 3: Implement `TriggerEdge.tsx`**

Create `frontend/src/components/Canvas/TriggerEdge.tsx`:

```tsx
import { BaseEdge, getBezierPath, type EdgeProps } from '@xyflow/react';

export function TriggerEdge(props: EdgeProps) {
  const [path] = getBezierPath({
    sourceX: props.sourceX,
    sourceY: props.sourceY,
    sourcePosition: props.sourcePosition,
    targetX: props.targetX,
    targetY: props.targetY,
    targetPosition: props.targetPosition,
  });

  return (
    <BaseEdge
      id={props.id}
      path={path}
      style={{
        stroke: '#22c55e',
        strokeDasharray: '6 4',
        strokeWidth: 2,
      }}
    />
  );
}
```

- [ ] **Step 4: Register in `FlowCanvas.tsx`**

Open `frontend/src/components/Canvas/FlowCanvas.tsx`. Find the `nodeTypes` registration (line 38-41). Update both maps:

```tsx
import { StartNode } from '../Nodes/StartNode';
import { TriggerEdge } from './TriggerEdge';

const nodeTypes: NodeTypes = {
  baseNode: BaseNode,
  presetNode: PresetNode,
  start: StartNode,  // NEW
};

const edgeTypes: EdgeTypes = {
  triggerEdge: TriggerEdge,  // NEW
};
```

Pass `edgeTypes={edgeTypes}` to the `<ReactFlow>` component if not already wired.

- [ ] **Step 5: Update `handleConnect` to mark trigger edges**

In `FlowCanvas.tsx`, find `handleConnect` (line 87-124). Add a branch: if the source handle is `trigger`, set the new edge's `type` to `triggerEdge` and `data.type` to `'trigger'`. Look for the `// Color the new edge by source port data type` comment and add **before** it:

```tsx
if (connection.sourceHandle === 'trigger') {
  const { setEdges } = useTabStore.getState();
  setEdges((edges) =>
    edges.map((e) =>
      e.source === connection.source &&
      e.target === connection.target &&
      e.sourceHandle === connection.sourceHandle
        ? { ...e, type: 'triggerEdge', data: { ...(e.data ?? {}), type: 'trigger' } }
        : e,
    ),
  );
  return; // skip the data-edge color logic
}
```

- [ ] **Step 6: Smoke test**

```bash
pnpm dev
```

Drag a Start node onto the canvas (you'll need to use the NodePalette — Task 12 makes Control discoverable; for now, you can manually add a node via the store or wait for Task 12). The Start node should render with a green pill.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/Nodes/StartNode.tsx frontend/src/components/Nodes/StartNode.module.css frontend/src/components/Canvas/TriggerEdge.tsx frontend/src/components/Canvas/FlowCanvas.tsx
git commit -m "feat(frontend): add StartNode component and trigger edge rendering"
```

---

### Task 12: Move `Control` category to first position

**Files:**
- Modify: `frontend/src/components/Sidebar/NodePalette.tsx:11`

- [ ] **Step 1: Read current `CATEGORY_ORDER`**

The current array (line 11) is:

```typescript
const CATEGORY_ORDER = ['Data', 'IO', 'CNN', 'Normalization', 'RNN', 'Transformer', 'RL', 'Training', 'Tensor Operations', 'Control', 'Utility'];
```

- [ ] **Step 2: Move `'Control'` to position 0**

Replace the line with:

```typescript
const CATEGORY_ORDER = ['Control', 'Data', 'IO', 'CNN', 'Normalization', 'RNN', 'Transformer', 'RL', 'Training', 'Tensor Operations', 'Utility'];
```

- [ ] **Step 3: Smoke test**

```bash
pnpm dev
```

Open the NodePalette sidebar. Verify `Control` appears as the first category, containing the `Start` node.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/Sidebar/NodePalette.tsx
git commit -m "feat(frontend): show Control category first in NodePalette"
```

---

### Task 13: Boost `Start` ranking in `QuickNodeSearch`

**Files:**
- Modify: `frontend/src/components/Canvas/QuickNodeSearch.tsx:32-48`

**Context:** Current ranking is just `includes` match on name+description, ordered as the definitions array yields them. Push the Start node to the front when the query is empty or matches "start".

- [ ] **Step 1: Read current ranking logic**

Read `frontend/src/components/Canvas/QuickNodeSearch.tsx` lines 32-48.

- [ ] **Step 2: Modify ranking**

Replace the search loop body with one that boosts Start:

```typescript
const results: SearchResult[] = (() => {
  const q = query.toLowerCase().trim();
  const items: SearchResult[] = [];

  for (const def of definitions) {
    if (!q || def.node_name.toLowerCase().includes(q) || def.description.toLowerCase().includes(q)) {
      items.push({ kind: 'node', def });
    }
  }
  for (const preset of presets) {
    if (!q || preset.preset_name.toLowerCase().includes(q) || preset.description.toLowerCase().includes(q)) {
      items.push({ kind: 'preset', preset });
    }
  }

  // Boost: Start node ranks first when query is empty or matches "start"
  if (!q || 'start'.includes(q)) {
    items.sort((a, b) => {
      const aIsStart = a.kind === 'node' && a.def.node_name === 'Start';
      const bIsStart = b.kind === 'node' && b.def.node_name === 'Start';
      if (aIsStart && !bIsStart) return -1;
      if (!aIsStart && bIsStart) return 1;
      return 0;
    });
  }

  return items.slice(0, 20);
})();
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Canvas/QuickNodeSearch.tsx
git commit -m "feat(frontend): boost Start node ranking in QuickNodeSearch"
```

---

## Phase 3: Auto Layout

### Task 14: `autoLayout.ts` — full implementation with unit tests

**Files:**
- Create: `frontend/src/utils/autoLayout.ts`
- Create: `frontend/src/utils/autoLayout.test.ts`

**Context:** The full layout algorithm: pickTargets (mode-based filter), findConnectedComponents (union-find), layoutComponentWithDagre (per-component LR), packIntoSwimLanes (stack components vertically). Selected mode preserves the centroid of the selection.

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/utils/autoLayout.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { autoLayout, type LayoutMode } from './autoLayout';
import type { Node, Edge } from '@xyflow/react';

function makeNode(id: string, x = 0, y = 0, isEntry = false): Node {
  return {
    id,
    position: { x, y },
    data: { id, type: 'Dataset', isEntryPoint: isEntry },
    type: 'baseNode',
    width: 200,
    height: 80,
  };
}

function makeEdge(id: string, source: string, target: string, type: 'data' | 'trigger' = 'data'): Edge {
  return { id, source, target, data: { type } };
}

describe('autoLayout', () => {
  it('linear chain: A→B→C→D produces strictly increasing X at same Y', () => {
    const nodes = [
      makeNode('A', 0, 0, true),
      makeNode('B', 0, 0),
      makeNode('C', 0, 0),
      makeNode('D', 0, 0),
    ];
    const edges = [
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'B', 'C'),
      makeEdge('e3', 'C', 'D'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const sorted = ['A', 'B', 'C', 'D'].map((id) => result.find((n) => n.id === id)!);
    // Strictly increasing X
    expect(sorted[0].position.x).toBeLessThan(sorted[1].position.x);
    expect(sorted[1].position.x).toBeLessThan(sorted[2].position.x);
    expect(sorted[2].position.x).toBeLessThan(sorted[3].position.x);
    // Same Y (within rounding)
    const ys = sorted.map((n) => n.position.y);
    expect(Math.max(...ys) - Math.min(...ys)).toBeLessThan(5);
  });

  it('diamond: A→B,A→C,B→D,C→D — B and C stack vertically at same X', () => {
    const nodes = ['A', 'B', 'C', 'D'].map((id) => makeNode(id, 0, 0, id === 'A'));
    const edges = [
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'A', 'C'),
      makeEdge('e3', 'B', 'D'),
      makeEdge('e4', 'C', 'D'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const B = result.find((n) => n.id === 'B')!;
    const C = result.find((n) => n.id === 'C')!;
    // B and C at the same X
    expect(Math.abs(B.position.x - C.position.x)).toBeLessThan(5);
    // Different Y
    expect(B.position.y).not.toBe(C.position.y);
  });

  it('two disconnected components → distinct Y bands', () => {
    const nodes = [
      makeNode('A1', 0, 0, true),
      makeNode('A2', 0, 0),
      makeNode('B1', 0, 0, true),
      makeNode('B2', 0, 0),
    ];
    const edges = [
      makeEdge('e1', 'A1', 'A2'),
      makeEdge('e2', 'B1', 'B2'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const A1y = result.find((n) => n.id === 'A1')!.position.y;
    const B1y = result.find((n) => n.id === 'B1')!.position.y;
    expect(A1y).not.toBe(B1y);
  });

  it('cycle A→B→C→A does not crash and produces valid coordinates', () => {
    const nodes = ['A', 'B', 'C'].map((id) => makeNode(id, 0, 0, id === 'A'));
    const edges = [
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'B', 'C'),
      makeEdge('e3', 'C', 'A'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    for (const n of result) {
      expect(Number.isFinite(n.position.x)).toBe(true);
      expect(Number.isFinite(n.position.y)).toBe(true);
    }
  });

  it('mode=experiments leaves draft components untouched', () => {
    const nodes = [
      makeNode('live1', 100, 100, true),
      makeNode('live2', 200, 100),
      makeNode('draft1', 500, 500),
      makeNode('draft2', 700, 500),
    ];
    const edges = [
      makeEdge('e1', 'live1', 'live2'),
      makeEdge('e2', 'draft1', 'draft2'),
    ];
    const result = autoLayout(nodes, edges, 'experiments');
    const draft1 = result.find((n) => n.id === 'draft1')!;
    const draft2 = result.find((n) => n.id === 'draft2')!;
    expect(draft1.position).toEqual({ x: 500, y: 500 });
    expect(draft2.position).toEqual({ x: 700, y: 500 });
  });

  it('mode=selected only moves selected nodes and preserves centroid', () => {
    const nodes = [
      makeNode('A', 100, 100, true),
      makeNode('B', 200, 100),
      makeNode('C', 300, 100),
      makeNode('untouched', 999, 999),
    ];
    const edges = [
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'B', 'C'),
    ];
    const selected = new Set(['A', 'B', 'C']);
    const result = autoLayout(nodes, edges, 'selected', selected);
    const untouched = result.find((n) => n.id === 'untouched')!;
    expect(untouched.position).toEqual({ x: 999, y: 999 });
    // Centroid of selection should be roughly preserved
    const beforeCentroid = { x: 200, y: 100 }; // (100+200+300)/3, 100
    const movedNodes = result.filter((n) => selected.has(n.id));
    const afterCentroid = {
      x: movedNodes.reduce((s, n) => s + n.position.x, 0) / movedNodes.length,
      y: movedNodes.reduce((s, n) => s + n.position.y, 0) / movedNodes.length,
    };
    expect(Math.abs(afterCentroid.x - beforeCentroid.x)).toBeLessThan(50);
    expect(Math.abs(afterCentroid.y - beforeCentroid.y)).toBeLessThan(50);
  });
});
```

- [ ] **Step 2: Run, watch fail**

```bash
pnpm vitest run src/utils/autoLayout.test.ts
```

Expected: failure (`autoLayout` does not exist).

- [ ] **Step 3: Implement `autoLayout.ts`**

Create `frontend/src/utils/autoLayout.ts`:

```typescript
import dagre from '@dagrejs/dagre';
import type { Node, Edge } from '@xyflow/react';

export type LayoutMode = 'experiments' | 'all' | 'selected';

const NODE_W = 200;
const NODE_H = 80;
const NODESEP = 40;
const RANKSEP = 80;
const LANE_GAP = 60;

function isEntryPoint(node: Node): boolean {
  return Boolean((node.data as any)?.isEntryPoint) || node.type === 'start';
}

function findConnectedComponents(targetIds: Set<string>, edges: Edge[]): string[][] {
  // Union-find on targetIds, treating ALL edges (data + trigger) as connecting.
  const parent = new Map<string, string>();
  for (const id of targetIds) parent.set(id, id);
  const find = (x: string): string => {
    let root = x;
    while (parent.get(root) !== root) root = parent.get(root)!;
    let cur = x;
    while (cur !== root) {
      const next = parent.get(cur)!;
      parent.set(cur, root);
      cur = next;
    }
    return root;
  };
  const union = (a: string, b: string) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  };
  for (const e of edges) {
    if (targetIds.has(e.source) && targetIds.has(e.target)) {
      union(e.source, e.target);
    }
  }
  const groups = new Map<string, string[]>();
  for (const id of targetIds) {
    const root = find(id);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root)!.push(id);
  }
  return Array.from(groups.values());
}

function layoutComponentWithDagre(
  componentNodeIds: string[],
  allNodes: Node[],
  allEdges: Edge[],
): Map<string, { x: number; y: number; width: number; height: number }> {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: 'LR', nodesep: NODESEP, ranksep: RANKSEP, ranker: 'network-simplex' });
  g.setDefaultEdgeLabel(() => ({}));

  const idSet = new Set(componentNodeIds);
  for (const id of componentNodeIds) {
    const node = allNodes.find((n) => n.id === id)!;
    const w = node.width ?? NODE_W;
    const h = node.height ?? NODE_H;
    g.setNode(id, { width: w, height: h });
  }
  for (const e of allEdges) {
    if (idSet.has(e.source) && idSet.has(e.target)) {
      g.setEdge(e.source, e.target);
    }
  }
  dagre.layout(g);

  const result = new Map<string, { x: number; y: number; width: number; height: number }>();
  for (const id of componentNodeIds) {
    const dn = g.node(id);
    // Dagre returns center coordinates; convert to top-left for React Flow.
    result.set(id, {
      x: dn.x - dn.width / 2,
      y: dn.y - dn.height / 2,
      width: dn.width,
      height: dn.height,
    });
  }
  return result;
}

interface LaidOutComponent {
  ids: string[];
  positions: Map<string, { x: number; y: number; width: number; height: number }>;
  hasEntryPoint: boolean;
  bounds: { minY: number; maxY: number };
}

function packIntoSwimLanes(
  components: LaidOutComponent[],
): Map<string, { x: number; y: number }> {
  // Sort: entry-pointed first, then drafts; within each group, larger first
  components.sort((a, b) => {
    if (a.hasEntryPoint !== b.hasEntryPoint) return a.hasEntryPoint ? -1 : 1;
    return b.ids.length - a.ids.length;
  });

  const finalPositions = new Map<string, { x: number; y: number }>();
  let currentY = 0;
  for (const comp of components) {
    const yOffset = currentY - comp.bounds.minY;
    let laneMaxY = -Infinity;
    for (const [id, pos] of comp.positions) {
      finalPositions.set(id, { x: pos.x, y: pos.y + yOffset });
      const bottom = pos.y + yOffset + pos.height;
      if (bottom > laneMaxY) laneMaxY = bottom;
    }
    currentY = laneMaxY + LANE_GAP;
  }
  return finalPositions;
}

function pickTargetIds(
  nodes: Node[],
  edges: Edge[],
  mode: LayoutMode,
  selectedIds?: Set<string>,
): Set<string> {
  if (mode === 'all') {
    return new Set(nodes.map((n) => n.id));
  }
  if (mode === 'selected') {
    return new Set(selectedIds ?? []);
  }
  // mode === 'experiments': only nodes in connected components that contain
  // at least one entry point
  const allComponents = findConnectedComponents(
    new Set(nodes.map((n) => n.id)),
    edges,
  );
  const targets = new Set<string>();
  for (const comp of allComponents) {
    const compNodes = comp.map((id) => nodes.find((n) => n.id === id)!);
    if (compNodes.some(isEntryPoint)) {
      for (const id of comp) targets.add(id);
    }
  }
  return targets;
}

export function autoLayout(
  nodes: Node[],
  edges: Edge[],
  mode: LayoutMode,
  selectedIds?: Set<string>,
): Node[] {
  const targetIds = pickTargetIds(nodes, edges, mode, selectedIds);
  if (targetIds.size === 0) return nodes;

  const componentIds = findConnectedComponents(targetIds, edges);

  // For 'selected' mode, record original centroid
  let originalCentroid: { x: number; y: number } | null = null;
  if (mode === 'selected' && selectedIds) {
    const sel = nodes.filter((n) => selectedIds.has(n.id));
    originalCentroid = {
      x: sel.reduce((s, n) => s + n.position.x + (n.width ?? NODE_W) / 2, 0) / sel.length,
      y: sel.reduce((s, n) => s + n.position.y + (n.height ?? NODE_H) / 2, 0) / sel.length,
    };
  }

  // Lay out each component independently
  const laidOut: LaidOutComponent[] = componentIds.map((ids) => {
    const positions = layoutComponentWithDagre(ids, nodes, edges);
    const ys = Array.from(positions.values()).map((p) => p.y);
    const heights = Array.from(positions.values()).map((p) => p.height);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys.map((y, i) => y + heights[i]));
    const compNodes = ids.map((id) => nodes.find((n) => n.id === id)!);
    return {
      ids,
      positions,
      hasEntryPoint: compNodes.some(isEntryPoint),
      bounds: { minY, maxY },
    };
  });

  let finalPositions = packIntoSwimLanes(laidOut);

  // Selected mode: shift result so the centroid matches the original
  if (mode === 'selected' && originalCentroid) {
    const sel = Array.from(finalPositions.entries());
    const newCentroid = {
      x: sel.reduce((s, [, p]) => s + p.x + NODE_W / 2, 0) / sel.length,
      y: sel.reduce((s, [, p]) => s + p.y + NODE_H / 2, 0) / sel.length,
    };
    const dx = originalCentroid.x - newCentroid.x;
    const dy = originalCentroid.y - newCentroid.y;
    finalPositions = new Map(
      Array.from(finalPositions.entries()).map(([id, p]) => [id, { x: p.x + dx, y: p.y + dy }]),
    );
  }

  // Build result: only target nodes get new positions; others unchanged
  return nodes.map((n) => {
    const newPos = finalPositions.get(n.id);
    if (!newPos) return n;
    return { ...n, position: newPos };
  });
}
```

- [ ] **Step 4: Run tests, watch them pass**

```bash
pnpm vitest run src/utils/autoLayout.test.ts
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/utils/autoLayout.ts frontend/src/utils/autoLayout.test.ts
git commit -m "feat(frontend): add autoLayout utility with Dagre and swim lanes"
```

---

### Task 15: `applyLayout` action + `lastLayoutMode` persisted setting

**Files:**
- Modify: `frontend/src/store/tabStore.ts`
- Modify: `frontend/src/store/uiStore.ts`

**Context:** `tabStore` already has an `undoStack: UndoSnapshot[]` and `undo()` action. `applyLayout` should push the current node positions onto the undo stack so `Ctrl+Z` reverts. `uiStore` has a localStorage pattern from `gridSnapEnabled`.

- [ ] **Step 1: Add `lastLayoutMode` to `uiStore`**

Edit `frontend/src/store/uiStore.ts`. After the `gridSnapEnabled` block, add:

```typescript
const LAYOUT_MODE_KEY = 'codefyui-last-layout-mode';

const loadLayoutMode = (): 'experiments' | 'all' | 'selected' => {
  const saved = localStorage.getItem(LAYOUT_MODE_KEY);
  if (saved === 'experiments' || saved === 'all' || saved === 'selected') return saved;
  return 'experiments';
};
```

In the store interface, add:

```typescript
lastLayoutMode: 'experiments' | 'all' | 'selected';
setLastLayoutMode: (mode: 'experiments' | 'all' | 'selected') => void;
```

In the store create() body:

```typescript
lastLayoutMode: loadLayoutMode(),
setLastLayoutMode: (mode) => {
  localStorage.setItem(LAYOUT_MODE_KEY, mode);
  set({ lastLayoutMode: mode });
},
```

- [ ] **Step 2: Add `applyLayout` to `tabStore`**

Open `frontend/src/store/tabStore.ts`. Add the import:

```typescript
import { autoLayout, type LayoutMode } from '../utils/autoLayout';
```

In the actions block, add:

```typescript
applyLayout: (mode: LayoutMode) => {
  const tabId = get().activeTabId;
  if (!tabId) return;
  set((state) => ({
    tabs: state.tabs.map((tab) => {
      if (tab.id !== tabId) return tab;
      // Snapshot for undo (uses existing UndoSnapshot pattern)
      const snapshot: UndoSnapshot = {
        nodes: tab.nodes.map((n) => ({ ...n, position: { ...n.position } })),
        edges: tab.edges,
      };
      const selectedIds = new Set(
        tab.nodes.filter((n) => n.selected).map((n) => n.id),
      );
      const newNodes = autoLayout(tab.nodes, tab.edges, mode, selectedIds);
      return {
        ...tab,
        nodes: newNodes,
        undoStack: [...tab.undoStack, snapshot],
        redoStack: [],
      };
    }),
  }));
},
```

Add to the interface:

```typescript
applyLayout: (mode: LayoutMode) => void;
```

**Note:** If `UndoSnapshot` shape is different from `{ nodes, edges }`, inspect the actual type and adapt accordingly. The existing `undo()` action shows the canonical snapshot shape.

- [ ] **Step 3: Smoke test**

```bash
pnpm dev
```

Open the dev console:

```javascript
useTabStore.getState().applyLayout('all')
```

Verify nodes reposition and `Ctrl+Z` reverts them.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/store/tabStore.ts frontend/src/store/uiStore.ts
git commit -m "feat(frontend): add applyLayout action with undo support"
```

---

### Task 16: Toolbar Auto Layout split button

**Files:**
- Modify: `frontend/src/components/Toolbar/Toolbar.tsx:509-575`
- Modify: `frontend/src/components/Toolbar/Toolbar.module.css`

**Context:** The right cluster of the toolbar already has toggles + the language selector dropdown. We add a split button: main button runs the last-used mode; caret opens a 3-item dropdown.

- [ ] **Step 1: Read current right-cluster structure**

Read `frontend/src/components/Toolbar/Toolbar.tsx` lines 509-575 to confirm the layout. Note the existing language dropdown pattern (lines ~545-565) — we'll mimic it.

- [ ] **Step 2: Add styles**

Append to `frontend/src/components/Toolbar/Toolbar.module.css`:

```css
.splitButton {
  display: inline-flex;
  border: 1px solid #444;
  border-radius: 4px;
  overflow: hidden;
  margin-left: 8px;
}

.splitButtonMain,
.splitButtonCaret {
  background: #2a2a2a;
  color: #e0e0e0;
  border: none;
  padding: 6px 10px;
  cursor: pointer;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.splitButtonMain:hover,
.splitButtonCaret:hover {
  background: #3a3a3a;
}

.splitButtonCaret {
  border-left: 1px solid #444;
  padding: 6px 6px;
}

.layoutDropdown {
  position: absolute;
  top: 100%;
  right: 0;
  background: #1f1f1f;
  border: 1px solid #444;
  border-radius: 4px;
  margin-top: 4px;
  min-width: 200px;
  z-index: 100;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
}

.layoutDropdownItem {
  padding: 8px 14px;
  cursor: pointer;
  color: #e0e0e0;
  font-size: 13px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.layoutDropdownItem:hover {
  background: #2a2a2a;
}

.layoutDropdownItemDisabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.layoutDropdownItemActive::after {
  content: "✓";
  color: #22c55e;
}
```

- [ ] **Step 3: Add the split button to `Toolbar.tsx`**

Open `frontend/src/components/Toolbar/Toolbar.tsx`. At the top, add imports:

```tsx
import { useState } from 'react';
import { useTabStore } from '../../store/tabStore';
import type { LayoutMode } from '../../utils/autoLayout';
```

Inside the component, before the return, add state:

```tsx
const [layoutMenuOpen, setLayoutMenuOpen] = useState(false);
const lastLayoutMode = useUIStore((s) => s.lastLayoutMode);
const setLastLayoutMode = useUIStore((s) => s.setLastLayoutMode);
const applyLayout = useTabStore((s) => s.applyLayout);
const selectedCount = useTabStore((s) => {
  const tab = s.tabs.find((t) => t.id === s.activeTabId);
  return tab?.nodes.filter((n) => n.selected).length ?? 0;
});

const runLayout = (mode: LayoutMode) => {
  setLastLayoutMode(mode);
  applyLayout(mode);
  setLayoutMenuOpen(false);
};
```

In the right cluster (around the language dropdown), add the split button BEFORE the language selector:

```tsx
<div className={styles.splitButton} style={{ position: 'relative' }}>
  <button
    className={styles.splitButtonMain}
    onClick={() => runLayout(lastLayoutMode)}
    title={t('toolbar.autoLayout')}
  >
    {t('toolbar.autoLayout')}
  </button>
  <button
    className={styles.splitButtonCaret}
    onClick={() => setLayoutMenuOpen((v) => !v)}
    aria-label="Layout mode"
  >
    ▾
  </button>
  {layoutMenuOpen && (
    <div className={styles.layoutDropdown}>
      <div
        className={`${styles.layoutDropdownItem} ${lastLayoutMode === 'experiments' ? styles.layoutDropdownItemActive : ''}`}
        onClick={() => runLayout('experiments')}
      >
        {t('toolbar.autoLayout.experiments')}
      </div>
      <div
        className={`${styles.layoutDropdownItem} ${lastLayoutMode === 'all' ? styles.layoutDropdownItemActive : ''}`}
        onClick={() => runLayout('all')}
      >
        {t('toolbar.autoLayout.all')}
      </div>
      <div
        className={`${styles.layoutDropdownItem} ${selectedCount === 0 ? styles.layoutDropdownItemDisabled : ''} ${lastLayoutMode === 'selected' ? styles.layoutDropdownItemActive : ''}`}
        onClick={() => {
          if (selectedCount > 0) runLayout('selected');
        }}
      >
        {t('toolbar.autoLayout.selected', { count: selectedCount })}
      </div>
    </div>
  )}
</div>
```

(The i18n keys are added in Task 20; for now they may render as the key itself — that's fine for this task.)

- [ ] **Step 4: Smoke test**

```bash
pnpm dev
```

Open the app, drag a few nodes onto the canvas, click the Auto Layout button — verify nodes rearrange. Click the caret — verify dropdown appears with 3 items.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/Toolbar/Toolbar.tsx frontend/src/components/Toolbar/Toolbar.module.css
git commit -m "feat(frontend): add Auto Layout split button to toolbar"
```

---

### Task 17: `Shift+L` keyboard shortcut

**Files:**
- Modify: `frontend/src/hooks/useKeyboardShortcuts.ts:15-35`

**Context:** Existing shortcut pattern: check modifier + key, prevent default, call store action. The shortcut runs the last-used layout mode.

- [ ] **Step 1: Read current shortcut handler**

Read `frontend/src/hooks/useKeyboardShortcuts.ts` to see the existing shortcut block structure.

- [ ] **Step 2: Add `Shift+L` binding**

In the keydown handler, after the existing shortcuts (e.g. after `Ctrl+Z`), add:

```typescript
if (e.shiftKey && !mod && e.key.toLowerCase() === 'l') {
  e.preventDefault();
  const mode = useUIStore.getState().lastLayoutMode;
  useTabStore.getState().applyLayout(mode);
  return;
}
```

Make sure `useUIStore` is imported at the top of the file if it isn't already.

- [ ] **Step 3: Smoke test**

```bash
pnpm dev
```

Press `Shift+L` in the canvas — verify nodes get laid out using the current `lastLayoutMode`.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/hooks/useKeyboardShortcuts.ts
git commit -m "feat(frontend): add Shift+L shortcut for auto layout"
```

---

## Phase 4: Validation + migration UX

### Task 18: `findEntryPoints` helper + `useGraphExecution` blocking

**Files:**
- Create: `frontend/src/utils/findEntryPoints.ts`
- Create: `frontend/src/utils/findEntryPoints.test.ts`
- Modify: `frontend/src/hooks/useGraphExecution.ts:107-120`

**Context:** Mirror the backend `find_entry_points` so the frontend can short-circuit a Run with a toast before sending anything to the backend. The toast tells the user what to do.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/utils/findEntryPoints.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { findEntryPoints } from './findEntryPoints';
import type { Node, Edge } from '@xyflow/react';

const node = (id: string, opts: { type?: string; isEntry?: boolean } = {}): Node => ({
  id,
  position: { x: 0, y: 0 },
  data: { id, type: opts.type ?? 'Dataset', isEntryPoint: opts.isEntry ?? false },
  type: 'baseNode',
});

const edge = (id: string, src: string, tgt: string, type: 'data' | 'trigger' = 'data'): Edge => ({
  id,
  source: src,
  target: tgt,
  data: { type },
});

describe('findEntryPoints', () => {
  it('returns explicit entry-pointed nodes', () => {
    const nodes = [node('a', { isEntry: true }), node('b')];
    expect(findEntryPoints(nodes, [])).toEqual(['a']);
  });

  it('returns Start nodes by type', () => {
    const nodes = [node('s', { type: 'Start' }), node('b')];
    expect(findEntryPoints(nodes, [])).toEqual(['s']);
  });

  it('returns nodes with incoming trigger edge', () => {
    const nodes = [node('s', { type: 'Start' }), node('ds')];
    const edges = [edge('e1', 's', 'ds', 'trigger')];
    const result = new Set(findEntryPoints(nodes, edges));
    expect(result).toEqual(new Set(['s', 'ds']));
  });

  it('returns empty when nothing is marked', () => {
    const nodes = [node('a'), node('b')];
    const edges = [edge('e1', 'a', 'b', 'data')];
    expect(findEntryPoints(nodes, edges)).toEqual([]);
  });
});
```

- [ ] **Step 2: Implement `findEntryPoints`**

Create `frontend/src/utils/findEntryPoints.ts`:

```typescript
import type { Node, Edge } from '@xyflow/react';

export function findEntryPoints(nodes: Node[], edges: Edge[]): string[] {
  const triggerTargets = new Set<string>();
  for (const e of edges) {
    if ((e.data as any)?.type === 'trigger') {
      triggerTargets.add(e.target);
    }
  }
  const result: string[] = [];
  for (const n of nodes) {
    const isMarker = Boolean((n.data as any)?.isEntryPoint);
    const isStartType = (n.data as any)?.type === 'Start' || n.type === 'start';
    const hasTriggerIn = triggerTargets.has(n.id);
    if (isMarker || isStartType || hasTriggerIn) {
      result.push(n.id);
    }
  }
  return result;
}
```

- [ ] **Step 3: Run tests**

```bash
pnpm vitest run src/utils/findEntryPoints.test.ts
```

Expected: 4 tests pass.

- [ ] **Step 4: Block run on no entry points**

Open `frontend/src/hooks/useGraphExecution.ts:107-120`. Find `const execute = useCallback(async () => {`. After getting the active tab and BEFORE the WS connect/serialization, add:

```typescript
import { findEntryPoints } from '../utils/findEntryPoints';
import { useToastStore } from '../store/toastStore';
import { useTranslation } from '../i18n/useTranslation';
// ... existing imports

const execute = useCallback(async () => {
  const tab = getActiveTab();
  // ... existing checks
  const entryIds = findEntryPoints(tab.nodes, tab.edges);
  if (entryIds.length === 0) {
    useToastStore.getState().addToast(
      t('execution.error.noEntryPoints'),
      'error',
    );
    return;
  }
  // ... continue with existing logic
}, [/* existing deps */]);
```

(The `t()` function needs to be in scope; if `useGraphExecution` isn't a React component already using `useTranslation`, hard-code the English string for now and Task 20 will replace it with the i18n key.)

- [ ] **Step 5: Smoke test**

```bash
pnpm dev
```

Drop a node without marking it, click Run — verify the toast appears and execution does NOT start.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/utils/findEntryPoints.ts frontend/src/utils/findEntryPoints.test.ts frontend/src/hooks/useGraphExecution.ts
git commit -m "feat(frontend): block run with toast when no entry points"
```

---

### Task 19: `MigrationModal` + on-load trigger

**Files:**
- Create: `frontend/src/components/shared/MigrationModal.tsx`
- Create: `frontend/src/components/shared/MigrationModal.module.css`
- Modify: `frontend/src/store/tabStore.ts` (or wherever graph load happens)
- Modify: `frontend/src/App.tsx` (to render the modal)

**Context:** When a graph is loaded that has no entry points and at least one node, show the modal. "Auto-mark all root nodes" sets `isEntryPoint=true` on every node with in-degree 0 (data edges only).

- [ ] **Step 1: Create the modal styles**

Create `frontend/src/components/shared/MigrationModal.module.css`:

```css
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}

.modal {
  background: #1f1f1f;
  border: 1px solid #444;
  border-radius: 8px;
  padding: 24px;
  max-width: 500px;
  color: #e0e0e0;
}

.title {
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 12px 0;
  color: #ffffff;
}

.body {
  font-size: 13px;
  line-height: 1.5;
  margin-bottom: 20px;
}

.buttons {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.button {
  background: #2a2a2a;
  color: #e0e0e0;
  border: 1px solid #444;
  border-radius: 4px;
  padding: 10px 14px;
  cursor: pointer;
  text-align: left;
  font-size: 13px;
}

.button:hover {
  background: #3a3a3a;
}

.buttonPrimary {
  background: linear-gradient(135deg, #16a34a, #22c55e);
  border-color: #22c55e;
  color: #ffffff;
  font-weight: 600;
}

.buttonPrimary:hover {
  background: linear-gradient(135deg, #15803d, #16a34a);
}
```

- [ ] **Step 2: Implement `MigrationModal.tsx`**

Create `frontend/src/components/shared/MigrationModal.tsx`:

```tsx
import styles from './MigrationModal.module.css';

export interface MigrationModalProps {
  open: boolean;
  onAutoMark: () => void;
  onOpenAsDraft: () => void;
  onCancel: () => void;
  t: (key: string) => string;
}

export function MigrationModal({ open, onAutoMark, onOpenAsDraft, onCancel, t }: MigrationModalProps) {
  if (!open) return null;
  return (
    <div className={styles.overlay} onClick={onCancel}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <h2 className={styles.title}>{t('migration.noEntryPoints.title')}</h2>
        <p className={styles.body}>{t('migration.noEntryPoints.body')}</p>
        <div className={styles.buttons}>
          <button className={`${styles.button} ${styles.buttonPrimary}`} onClick={onAutoMark}>
            {t('migration.autoMark')}
          </button>
          <button className={styles.button} onClick={onOpenAsDraft}>
            {t('migration.openAsDraft')}
          </button>
          <button className={styles.button} onClick={onCancel}>
            {t('migration.cancel')}
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Add `markAllRootsAsEntryPoints` action to `tabStore`**

Add to the store:

```typescript
markAllRootsAsEntryPoints: () => {
  const tabId = get().activeTabId;
  if (!tabId) return;
  set((state) => ({
    tabs: state.tabs.map((tab) => {
      if (tab.id !== tabId) return tab;
      // Find data-roots: nodes with no incoming data edge
      const targetIds = new Set(
        tab.edges
          .filter((e) => ((e.data as any)?.type ?? 'data') === 'data')
          .map((e) => e.target),
      );
      return {
        ...tab,
        nodes: tab.nodes.map((n) => {
          if (targetIds.has(n.id)) return n;
          return { ...n, data: { ...n.data, isEntryPoint: true } };
        }),
      };
    }),
  }));
},
```

- [ ] **Step 4: Trigger modal on graph load**

Find where graphs are loaded (probably `loadGraphFromJson` or similar in `tabStore.ts`). After load, check `findEntryPoints(nodes, edges).length === 0 && nodes.length > 0` and set a flag in `uiStore`:

```typescript
import { findEntryPoints } from '../utils/findEntryPoints';
// ... in the load action, after nodes/edges are set:
const entries = findEntryPoints(loadedNodes, loadedEdges);
if (entries.length === 0 && loadedNodes.length > 0) {
  useUIStore.getState().setMigrationModalOpen(true);
}
```

Add `migrationModalOpen` and `setMigrationModalOpen` to `uiStore.ts`:

```typescript
migrationModalOpen: false,
setMigrationModalOpen: (open: boolean) => set({ migrationModalOpen: open }),
```

- [ ] **Step 5: Render the modal in `App.tsx`**

Open `frontend/src/App.tsx`. Add the modal at the top level:

```tsx
import { MigrationModal } from './components/shared/MigrationModal';
import { useUIStore } from './store/uiStore';
import { useTabStore } from './store/tabStore';
import { useTranslation } from './i18n/useTranslation';

// inside App component:
const migrationOpen = useUIStore((s) => s.migrationModalOpen);
const closeModal = useUIStore((s) => s.setMigrationModalOpen);
const markAllRoots = useTabStore((s) => s.markAllRootsAsEntryPoints);
const { t } = useTranslation();

// in JSX:
<MigrationModal
  open={migrationOpen}
  onAutoMark={() => {
    markAllRoots();
    closeModal(false);
  }}
  onOpenAsDraft={() => closeModal(false)}
  onCancel={() => closeModal(false)}
  t={t}
/>
```

- [ ] **Step 6: Smoke test**

```bash
pnpm dev
```

Manually load `examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json` (the unmigrated version — this works because we haven't run the migration script yet). Verify the modal appears. Click "Auto-mark" and verify the Dataset node gets the green outline.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/shared/MigrationModal.tsx frontend/src/components/shared/MigrationModal.module.css frontend/src/store/tabStore.ts frontend/src/store/uiStore.ts frontend/src/App.tsx
git commit -m "feat(frontend): add migration modal for graphs without entry points"
```

---

## Phase 5: i18n

### Task 20: Add i18n strings (en + zh-TW)

**Files:**
- Modify: `frontend/src/i18n/locales/en.ts`
- Modify: `frontend/src/i18n/locales/zh-TW.ts`

- [ ] **Step 1: Add English strings**

Open `frontend/src/i18n/locales/en.ts`. Add these keys (place each in the appropriate section):

```typescript
// Toolbar section
'toolbar.autoLayout': 'Auto Layout',
'toolbar.autoLayout.experiments': 'Layout Experiments',
'toolbar.autoLayout.all': 'Layout All',
'toolbar.autoLayout.selected': 'Layout Selected ({count})',

// Node context menu section
'node.contextMenu.setAsEntryPoint': 'Set as Entry Point',
'node.contextMenu.removeEntryPoint': 'Remove Entry Point',

// Migration modal section
'migration.noEntryPoints.title': 'No Entry Points Defined',
'migration.noEntryPoints.body': 'This graph has no entry points. Without entry points, the graph cannot be executed. Choose how to handle it:',
'migration.autoMark': 'Auto-mark all root nodes as entry points',
'migration.openAsDraft': 'Open as draft (I\'ll mark them later)',
'migration.cancel': 'Cancel',

// Execution errors
'execution.error.noEntryPoints': 'No entry points defined. Right-click a root node and select "Set as Entry Point", or drag a Start node from the palette.',

// Node palette
'palette.category.control': 'Control',
'palette.start.description': 'Marks an execution entry point. Connect to the first node of a script.',
```

- [ ] **Step 2: Add Traditional Chinese strings**

Open `frontend/src/i18n/locales/zh-TW.ts`. Add the same keys with Chinese values:

```typescript
'toolbar.autoLayout': '自動排版',
'toolbar.autoLayout.experiments': '排版實驗',
'toolbar.autoLayout.all': '排版全部',
'toolbar.autoLayout.selected': '排版選取 ({count})',

'node.contextMenu.setAsEntryPoint': '設為起始節點',
'node.contextMenu.removeEntryPoint': '取消起始節點',

'migration.noEntryPoints.title': '尚未定義起始節點',
'migration.noEntryPoints.body': '此圖沒有定義任何起始節點。沒有起始節點就無法執行。請選擇處理方式：',
'migration.autoMark': '自動將所有根節點標記為起始節點',
'migration.openAsDraft': '當作草稿開啟（稍後再標記）',
'migration.cancel': '取消',

'execution.error.noEntryPoints': '尚未定義起始節點。請在根節點上按右鍵選擇「設為起始節點」，或從面板拖曳一個 Start 節點。',

'palette.category.control': '控制',
'palette.start.description': '標記執行的起點。連接到你想執行的第一個節點。',
```

- [ ] **Step 3: Verify the `{count}` interpolation works**

Check if the project's `t()` function supports `{count}` interpolation. Look at existing usages of `t()` with interpolation. If it doesn't (e.g. uses different syntax like `%s`), adapt the keys accordingly.

If the `t` helper doesn't support interpolation, change the key to `'toolbar.autoLayout.selected': 'Layout Selected'` and append the count in JSX:

```tsx
{t('toolbar.autoLayout.selected')} ({selectedCount})
```

- [ ] **Step 4: Smoke test**

```bash
pnpm dev
```

Verify the new strings show up in the UI in both English and Traditional Chinese (toggle via the language selector).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/i18n/locales/en.ts frontend/src/i18n/locales/zh-TW.ts
git commit -m "i18n: add strings for entry points and auto layout"
```

---

## Phase 6: Example migration

### Task 21: Migration script + run on 16 example graphs

**Files:**
- Create: `scripts/migrate_entry_points.py`
- Modify: `examples/**/graph.json` (×16, automated)

**Context:** A small Python helper that walks `examples/**/graph.json`, identifies data-roots (nodes with no incoming data edge), and sets `data.isEntryPoint = true` on each. Then run it.

- [ ] **Step 1: Create the migration script**

Create `scripts/migrate_entry_points.py`:

```python
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
```

- [ ] **Step 2: Run the migration**

From the repository root:

```bash
python scripts/migrate_entry_points.py
```

Expected output: 16 lines of `MODIFIED examples/...` (one per file), then `Done: 16/16 graphs updated.`

- [ ] **Step 3: Verify one graph manually**

Read `examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json` and confirm the root node (the one without any incoming edges) now has `"isEntryPoint": true` in its `data`.

- [ ] **Step 4: Run idempotency check**

Run the script a second time:

```bash
python scripts/migrate_entry_points.py
```

Expected: 16 lines of `skip     examples/...`, `Done: 0/16 graphs updated.`

- [ ] **Step 5: Boot the backend and verify each example loads**

```bash
cd backend && uv run uvicorn app.main:app --reload &
```

Then in another terminal:

```bash
for f in examples/Usage_Example/CNN-MNIST/*/graph.json examples/Model_Architecture/*/graph.json; do
  echo "=== $f ==="
  curl -s -X POST http://localhost:8000/api/graph/validate \
    -H "Content-Type: application/json" \
    -d "@$f" | python -m json.tool
done
```

Expected: every response has `"valid": true`. If any are invalid, inspect the error and fix the underlying issue (likely a missing entry point on a non-obvious root, or an existing structural issue unrelated to this migration).

Stop the backend (`kill %1` or Ctrl+C the uvicorn process).

- [ ] **Step 6: Commit**

```bash
git add scripts/migrate_entry_points.py examples/
git commit -m "feat(examples): migrate all 16 example graphs to use entry points"
```

---

## Phase 7: Final verification

### Task 22: Run full backend test suite

- [ ] **Step 1: Run pytest**

From `backend/`:

```bash
uv run pytest -v
```

Expected: all tests pass. If any fail, fix root cause before proceeding (do NOT mark this task complete with failing tests).

- [ ] **Step 2: Note the test count**

Record the total test count and any tests that took unusually long. No commit needed for this task — it's a verification gate.

---

### Task 23: Run full frontend test suite

- [ ] **Step 1: Run vitest**

From `frontend/`:

```bash
pnpm vitest run
```

Expected: all tests pass.

- [ ] **Step 2: Note the count**

Record total test count. No commit.

---

### Task 24: Manual Chrome verification

**Context:** The user has explicitly required browser-based verification. Use the `mcp__claude-in-chrome__*` MCP tools throughout this task. Capture a GIF of the auto-layout flow.

- [ ] **Step 1: Boot dev environment**

In one terminal:

```bash
cd backend && uv run uvicorn app.main:app --reload --port 8000
```

In another:

```bash
cd frontend && pnpm dev
```

Confirm the frontend is at `http://localhost:5173` (or whatever Vite reports).

- [ ] **Step 2: Open in Chrome via MCP**

```
mcp__claude-in-chrome__tabs_context_mcp  → check existing tabs
mcp__claude-in-chrome__tabs_create_mcp  → create new tab pointing to http://localhost:5173
```

- [ ] **Step 3: Verification checklist (run each step in Chrome)**

For each item, perform the action via `mcp__claude-in-chrome__find` / `mcp__claude-in-chrome__form_input` / `mcp__claude-in-chrome__computer` and confirm via `mcp__claude-in-chrome__read_page` or screenshot.

  1. **Entry point via marker**: drag a Dataset node onto the canvas. Right-click it. Verify `Set as Entry Point` appears. Click it. Verify the green outline + flag badge appears.
  2. **Entry point via Start node**: open NodePalette, verify `Control > Start` is at the top. Drag a Start node onto the canvas — verify it has the green outline. Drag a connection from its trigger output to a fresh Dataset node — verify the edge is green dashed.
  3. **Execute experiment vs. draft**: build two disconnected chains, mark only one. Click Run. Verify only the marked chain executes (check log panel). Mark the other, run again, verify both execute.
  4. **No entry point validation**: remove all entry points. Click Run. Verify a toast appears with the no-entry-points message and execution does not start.
  5. **Auto Layout — Experiments mode**: build a messy entry-pointed chain alongside a draft. Click `Auto Layout`. Verify only the entry-pointed chain rearranges.
  6. **Auto Layout — All mode**: from the dropdown, pick `Layout All`. Verify both chains rearrange into stacked swim lanes (entry-pointed first).
  7. **Auto Layout — Selected mode**: select 3 nodes from a chain. From the dropdown, pick `Layout Selected (3)`. Verify only those 3 nodes move; rest stay put.
  8. **Auto Layout with cycle**: create a temporary cycle (edge from a node back to its ancestor). Click `Auto Layout`. Verify it does not crash, nodes are positioned, back-edge renders as a curved loop.
  9. **Migration modal**: stash the migration commit (`git stash`), reload, load `examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json`. Verify the migration modal appears. Click `Auto-mark all root nodes`. Verify the Dataset node has the entry-point outline. Verify Run works. Restore (`git stash pop`).
  10. **Migrated example graphs**: load `TrainCNN-MNIST` (now migrated). Verify no migration modal. Run end-to-end and verify training progresses for at least 1 epoch. Spot-check ResNet and ViT examples — load + verify execution starts without validation errors.
  11. **Keyboard shortcut**: press `Shift+L` — verify it triggers the last-used layout mode.
  12. **Undo**: run Auto Layout, press `Ctrl+Z` — verify positions revert.

- [ ] **Step 4: Record a GIF of step 5 (Auto Layout — Experiments mode)**

Use `mcp__claude-in-chrome__gif_creator` to record the messy → clean transition. Save with a descriptive name (`auto_layout_demo.gif`).

- [ ] **Step 5: Stop dev servers**

Ctrl+C the `pnpm dev` and `uvicorn` processes.

- [ ] **Step 6: Final commit (only if any docs/cleanup needed)**

If the GIF should be committed somewhere (e.g. `docs/`), commit it. Otherwise no final commit is required for this task — it's pure verification.

```bash
# only if a GIF or docs file was created
git add docs/auto_layout_demo.gif
git commit -m "docs: add auto layout demo gif"
```

---

## Done

All 24 tasks complete. Entry points and auto layout are wired end-to-end with full backend gating, frontend visuals, dagre-based layout, migration support, and verified manually in Chrome.
