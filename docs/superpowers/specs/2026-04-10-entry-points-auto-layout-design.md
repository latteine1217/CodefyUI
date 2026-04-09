# Entry Points & Auto Layout

**Date:** 2026-04-10
**Status:** Approved (brainstorming)
**Owner:** treeleaves30760

## Goal

Solve two related problems in CodefyUI's GUI editor:

1. **Messy canvas after editing** — Add an "Auto Layout" feature that arranges
   nodes in a clean horizontal left-to-right flow with vertical stacking for
   siblings, supports multiple disconnected subgraphs (swim lanes), and degrades
   gracefully when the user has temporarily created a cyclic connection.

2. **Ambiguous execution start** — Today the backend executes *every* node with
   in-degree 0 in parallel. With multiple disconnected subgraphs on the canvas
   (e.g. an in-progress draft sitting next to a real experiment), the runs
   "fight" over GPU memory, output panels, and logs. Introduce an explicit
   **Entry Point** concept so the user declares which subgraphs are live.

## Background

### Current execution model

- `backend/app/core/graph_engine.py` uses Kahn's algorithm for topological
  sort. `validate_graph()` rejects cycles outright with the error
  `"Graph contains a cycle"`.
- `topological_levels()` groups nodes by DAG level; level-0 nodes (every node
  with in-degree 0) execute concurrently, then their descendants.
- There is no notion of an "entry node" — execution implicitly starts from
  every root.

### Current canvas

- Frontend uses **React Flow v12** (`@xyflow/react`). Node positions are
  manually placed and stored in `node.position: {x, y}`.
- No layout library is installed (`frontend/package.json` has no `dagre` or
  `elkjs`).
- The toolbar (`frontend/src/components/Toolbar/Toolbar.tsx`) has a right
  cluster suitable for an "Auto Layout" action button.

### Example graphs in the repo

There are **16** example graphs under `examples/` (2 in `Usage_Example/`,
14 in `Model_Architecture/`). All of them currently rely on the implicit
"every root is a start" behaviour and will need entry-point markers.

## Decisions (from brainstorming)

| # | Decision | Rationale |
|---|---|---|
| Q1 | **Multiple disconnected subgraphs**: introduce explicit Entry Points (option A from brainstorming). Subgraphs without an entry point are treated as drafts and skipped during execution. | Users want to keep draft subgraphs alongside real experiments without them fighting for GPU/output. Explicit declaration is clearer than auto-detection. |
| Q2 | **Entry Point representation**: combine option B (markable root nodes) **and** option C (dedicated `start` node type). Both produce the same execution semantic. Both share the same green outline + flag badge visual. | Marker is cheap for power users; Start node is discoverable for newcomers. Visual consistency keeps the mental model unified. Like Scratch's "When Flag Clicked" hat block. |
| Q3 | **Layout direction**: horizontal Left-to-Right, with vertical stacking for siblings at the same topological depth. | Standard for ML workflows (TF Graph, PyTorch Lightning summary); React Flow's default bezier edges look cleanest horizontal. |
| Q4 | **Multi-component layout**: Swim lanes — one horizontal lane per connected component, stacked top-to-bottom. | One experiment per row matches the user's mental model of running parallel experiments side-by-side; lines up with the entry-point semantic (each swim lane = one execution unit). |
| Q5 | **Cycle handling in layout**: visual only — Dagre breaks cycles for layout, the back-edge is rendered as a regular React Flow edge that visually loops back. The backend continues to reject cycles at execution time. | Auto-layout must work even on temporarily-broken graphs (mid-edit), and visualising the back-edge actually helps users locate the cycle they accidentally created. |
| Q6 | **Layout scope**: three modes via dropdown — Layout Experiments (default, only entry-pointed components), Layout All, Layout Selected. The button click uses the last-used mode; the caret opens the menu. | Maximum flexibility without UI clutter. Last-used mode persistence keeps the common case one-click. |
| Q7 | **Trigger edge in-degree**: edges from `start` nodes (type `trigger`) do **not** count toward a node's in-degree for execution-order purposes, but **do** count for connected-component grouping. | The Start node is a marker, not a data dependency; it should not change the topological execution order, but its visual connection to the downstream node should establish the "this Start belongs to this component" relationship. |
| Q8 | **Strict no-entry-point validation**: a graph with no entry points fails validation and refuses to execute. **However**, when loading an old graph file without entry points, the UI offers a one-click "auto-mark all root nodes as entry points" migration. | Strict at execution time keeps the semantic clean. Migration prompt makes the upgrade path zero-friction for existing users. |

## Architecture overview

```
                    Frontend                        Backend
   ┌───────────────────────────────────┐   ┌─────────────────────────┐
   │  Canvas (React Flow)              │   │ graph_engine.py         │
   │   ├─ BaseNode + .entryPoint       │   │  ├─ find_entry_points() │
   │   │   modifier (green outline)    │   │  ├─ reachable_from_     │
   │   ├─ StartNode (new type)         │   │  │   entry_points()     │
   │   └─ Trigger edge style (dashed)  │   │  ├─ validate_graph()    │
   │                                   │   │  │   (entry-point check)│
   │  Toolbar                          │   │  └─ execute_graph()     │
   │   └─ Auto Layout split button     │   │     (skips unreachable) │
   │                                   │   │                         │
   │  utils/autoLayout.ts (Dagre)      │   │ schemas/models.py       │
   │   ├─ pickTargets(mode)            │   │  ├─ NodeData.isEntry-   │
   │   ├─ findConnectedComponents()    │   │  │   Point: bool        │
   │   ├─ layoutComponentWithDagre()   │   │  └─ EdgeData.type:      │
   │   └─ packIntoSwimLanes()          │   │      'data' | 'trigger' │
   │                                   │   │                         │
   │  MigrationModal (load old graph)  │   │ nodes/control/start.py  │
   │                                   │   │  (no-op marker node)    │
   └───────────────────────────────────┘   └─────────────────────────┘
```

## Data model changes

### Backend

`backend/app/schemas/models.py`:

```python
class NodeData(BaseModel):
    # ... existing fields
    isEntryPoint: bool = False  # NEW

class EdgeData(BaseModel):
    # ... existing fields
    type: Literal["data", "trigger"] = "data"  # NEW; default "data" preserves old graphs
```

### Frontend

`frontend/src/store/tabStore.ts` — the React Flow `Node<NodeData>` type already
spreads from backend `NodeData`, so adding `isEntryPoint` backend-side propagates
automatically. Same for `Edge.data.type`.

## Backend changes

### `backend/app/core/graph_engine.py`

**New helpers:**

```python
def find_entry_points(nodes: list[NodeData], edges: list[EdgeData]) -> list[str]:
    """Return ids of nodes that are entry points: either explicitly marked
    via isEntryPoint=True, or with at least one incoming trigger edge."""

def reachable_from_entry_points(
    entry_ids: list[str],
    edges: list[EdgeData]
) -> set[str]:
    """BFS forward through DATA edges only; trigger edges are not traversed
    because they are markers, not data dependencies. The entry point itself
    and all data-downstream nodes are included."""
```

**Modified `validate_graph()`:**

1. Run existing structural validation (missing handles, type mismatches, etc.).
2. Find entry points; if `len(entry_ids) == 0` → raise
   `"Graph has no entry points. Mark a root node as an entry point or add a Start node."`
3. Compute reachable set; let `executable_nodes = reachable_set`.
4. Run cycle detection **only on the executable subgraph**, not the whole graph.
   This way drafts with temporary cycles don't break a real run.
5. Validate that no entry-point node has any incoming **data** edge. This
   applies whether the node is an entry via `isEntryPoint=True` *or* via an
   incoming trigger edge from a Start node — entry points must be data-roots
   in both cases. Matches Scratch's hat-block semantic: nothing flows *into*
   the start of a script.

**Modified `topological_levels()` and `execute_graph()`:**

- Operate on `executable_nodes` only — drafts are skipped silently.
- When computing in-degree, **exclude** edges of type `trigger`.
- Start nodes appear in `executable_nodes` (they are entry points by virtue of
  their trigger output) but their `execute()` is a no-op that returns no
  outputs. They occupy a topological level 0 slot but contribute nothing to
  downstream.

### `backend/app/nodes/control/start_node.py` (new)

```python
class StartNode(BaseNode):
    NODE_TYPE = "start"
    DISPLAY_NAME = "Start"
    CATEGORY = "Control"
    DESCRIPTION = "Marks an execution entry point. Connect this to the first node of a script you want to run."

    INPUTS = []  # no inputs
    OUTPUTS = [NodeOutput(name="trigger", type=DataType.TRIGGER)]
    PARAMS = []

    def execute(self, ctx) -> dict:
        return {}  # no-op; downstream uses data edges, not trigger edges
```

A new `DataType.TRIGGER` value is added to the existing `DataType` enum so
trigger ports can only connect to other trigger ports (this guards against
accidental connection to a real data input).

### `backend/app/api/routes_nodes.py`

Auto-discovery already picks up new files in `backend/app/nodes/**`, so the
new `start_node.py` is registered automatically. Verify a `Control` category
shows up in `/api/nodes`.

## Frontend changes

### Visual: entry point outline

`frontend/src/components/Nodes/BaseNode.module.css` — add an `.entryPoint`
modifier class:

```css
.entryPoint {
  outline: 2px solid;
  outline-offset: 2px;
  /* gradient via border-image or simply solid for first cut */
  outline-color: #22c55e;
  box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.18);
}

.entryPoint::before {
  content: ""; /* SVG flag icon via background-image */
  position: absolute;
  top: -8px;
  left: -8px;
  width: 16px;
  height: 16px;
  background: url("./flag-icon.svg") no-repeat center;
}
```

`BaseNode.tsx` — apply `styles.entryPoint` when `data.isEntryPoint === true`.

### New: StartNode component

`frontend/src/components/Nodes/StartNode.tsx` (new file):

- Render as a smaller pill-shaped node (~ 80×40px) with a flag SVG and "Start"
  label
- Single output handle on the right, type `trigger`, rendered as a small green
  triangle (CSS variant of the existing handle)
- Always shows the `.entryPoint` outline (it is permanently an entry point)
- Registered in `FlowCanvas.tsx` `nodeTypes` map under `start`

### New: trigger edge style

`frontend/src/components/Canvas/FlowCanvas.tsx` — register a custom edge type
`triggerEdge` with:

- `stroke: #22c55e`
- `strokeDasharray: "6 4"`
- `strokeWidth: 2`
- No edge label

When the user drags from a Start node's trigger handle, the resulting edge
gets `data.type = "trigger"` and the `triggerEdge` type.

### Right-click "Set as Entry Point"

`BaseNode.tsx` — add `onContextMenu` handler:

1. Compute in-degree from current edges (filter out trigger edges) — only show
   the menu item for nodes with in-degree 0.
2. Render a small menu (reuse existing context-menu pattern if any; otherwise
   absolute-positioned div anchored to cursor).
3. Items: `Set as Entry Point` ↔ `Remove Entry Point` (toggle based on current
   state).
4. Action calls a new `tabStore` mutation `toggleEntryPoint(nodeId)`.

### NodePalette: Control category

`frontend/src/components/Sidebar/NodePalette.tsx` — when category list is
built, ensure `Control` (containing `Start`) is rendered **first** so it's the
most visible. Backend's `/api/nodes` already includes it after registration.

### QuickNodeSearch

`frontend/src/components/Canvas/QuickNodeSearch.tsx` — when the search query
is empty or matches `start`, the Start node ranks first in suggestions.

### Auto Layout button

`frontend/src/components/Toolbar/Toolbar.tsx` — add a split button in the
right cluster (after the existing toggle group). UI:

```
[ ⊞ Auto Layout ] [ ▾ ]
```

- Click main: run last-used mode (default `experiments`)
- Click caret: dropdown with three radio items
  - `Layout Experiments` (default)
  - `Layout All`
  - `Layout Selected (N)` — disabled when no selection, shows count when active

### `frontend/src/utils/autoLayout.ts` (new)

```typescript
import dagre from '@dagrejs/dagre';
import type { Node, Edge } from '@xyflow/react';

export type LayoutMode = 'experiments' | 'all' | 'selected';

const NODE_W = 200;       // approximate; refined per node type
const NODE_H = 80;
const NODESEP = 40;       // vertical gap between siblings
const RANKSEP = 80;       // horizontal gap between layers
const LANE_GAP = 60;      // vertical gap between swim lanes

export function autoLayout(
  nodes: Node[],
  edges: Edge[],
  mode: LayoutMode,
  selectedIds?: Set<string>,
): Node[] {
  const targetIds = pickTargets(nodes, edges, mode, selectedIds);
  const components = findConnectedComponents(targetIds, edges);

  // Lay out each component independently with Dagre
  const laidOut = components.map(comp => layoutComponentWithDagre(comp, nodes, edges));

  // Sort: components with entry points first, then drafts; within each group,
  // larger components first (visual weight)
  laidOut.sort(byEntryPointThenSize);

  // Stack as swim lanes
  return packIntoSwimLanes(laidOut, nodes, targetIds);
}
```

**Connected components:** union-find on `targetIds`, treating *all* edges
(both data and trigger) as connecting. This ensures Start nodes group with
their downstream targets.

**Per-component Dagre call:**

```typescript
function layoutComponentWithDagre(componentNodeIds: string[], allNodes: Node[], allEdges: Edge[]) {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: 'LR', nodesep: NODESEP, ranksep: RANKSEP, ranker: 'network-simplex' });
  g.setDefaultEdgeLabel(() => ({}));

  for (const id of componentNodeIds) {
    const node = allNodes.find(n => n.id === id)!;
    g.setNode(id, { width: node.width ?? NODE_W, height: node.height ?? NODE_H });
  }
  for (const e of allEdges) {
    if (componentNodeIds.includes(e.source) && componentNodeIds.includes(e.target)) {
      g.setEdge(e.source, e.target);
    }
  }
  dagre.layout(g);
  return componentNodeIds.map(id => ({ id, ...g.node(id) }));
}
```

Dagre returns center coordinates; convert to React Flow's top-left origin via
`x = center.x - width/2`, `y = center.y - height/2`.

**Swim lane packing:** for each component (sorted), compute its bounding box,
shift it so its top edge is at `currentY`, then `currentY += comp.height + LANE_GAP`.
All components share the same `x = 0` origin.

**Selected mode preserves anchor:** record the centroid of the original
selection, run layout, then translate the result so the new centroid matches
the old one. Non-selected nodes are untouched.

### Store / undo

`frontend/src/store/tabStore.ts` — add:

```typescript
applyLayout(mode: LayoutMode): void
undoLastLayout(): void          // restores from snapshot
```

`applyLayout` snapshots `nodes.map(n => ({ id: n.id, position: n.position }))`
into `state.lastLayoutSnapshot` *before* mutating. `undoLastLayout` restores
the snapshot. We are **not** building a general-purpose undo/redo system in
this spec — only this single-step layout undo. (Out of scope for this spec is
deciding on a full undo/redo library.)

`frontend/src/store/uiStore.ts` — add `lastLayoutMode: LayoutMode = 'experiments'`,
persisted to localStorage.

`frontend/src/hooks/useKeyboardShortcuts.ts` — bind `Shift+L` to
`applyLayout(uiStore.lastLayoutMode)`.

### Migration modal for old graphs

`frontend/src/components/shared/MigrationModal.tsx` (new):

When a graph is loaded (file open, example load) and `find_entry_points()`
returns empty AND there is at least one node, show:

```
This graph has no entry points defined.

Without entry points, the graph cannot be executed. Choose how to handle it:

  [ Auto-mark all root nodes as entry points ]   ← recommended, default focus
  [ Open as draft (I'll mark them later)      ]
  [ Cancel                                     ]
```

"Auto-mark" finds every node with in-degree 0 (data edges only) and sets
`isEntryPoint = true`. "Open as draft" loads the graph as-is; the user must
mark entries before running.

### i18n

Add new strings to both `frontend/src/i18n/locales/en.ts` and `zh-TW.ts`:

- `toolbar.autoLayout`, `toolbar.autoLayout.experiments`, `toolbar.autoLayout.all`, `toolbar.autoLayout.selected`
- `node.contextMenu.setAsEntryPoint`, `node.contextMenu.removeEntryPoint`
- `migration.noEntryPoints.title`, `migration.noEntryPoints.body`, `migration.autoMark`, `migration.openAsDraft`
- `execution.error.noEntryPoints`
- `nodePalette.category.control`, `nodePalette.start.description`

## Example graph migration

All 16 example graphs need entry points:

```
examples/Usage_Example/CNN-MNIST/InferenceCNN-MNIST/graph.json
examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
examples/Model_Architecture/BERT-Encoder-Transformer/graph.json
examples/Model_Architecture/BiGRU-SpeechRecognition-RNN/graph.json
examples/Model_Architecture/ConvNeXt-CNN/graph.json
examples/Model_Architecture/DQN-Atari-RL/graph.json
examples/Model_Architecture/DiT-Diffusion-Transformer/graph.json
examples/Model_Architecture/EfficientNet-CNN/graph.json
examples/Model_Architecture/GPT-DecoderOnly-Transformer/graph.json
examples/Model_Architecture/LLaMA-Decoder-Transformer/graph.json
examples/Model_Architecture/PPO-Robotics-RL/graph.json
examples/Model_Architecture/ResNet-SkipConnection-CNN/graph.json
examples/Model_Architecture/Seq2Seq-Attention-RNN/graph.json
examples/Model_Architecture/SwinTransformer-Transformer/graph.json
examples/Model_Architecture/TimeSeries-LSTM-RNN/graph.json
examples/Model_Architecture/ViT-ImageClassifier-Transformer/graph.json
```

**Migration approach:** for each graph, find every node with in-degree 0 in
the data-edge graph (these are guaranteed to be `Dataset` / `DataLoader` /
`HuggingFaceDatasetNode` / `KaggleDatasetNode` based on the current example
shapes) and set `data.isEntryPoint: true`. Use the inline marker approach for
simplicity — no need to add Start nodes to existing examples (Start nodes are
introduced as a new pattern for users authoring fresh graphs). The migration
can be scripted: a small Python helper that walks `examples/**/graph.json`,
identifies roots, and sets the flag in place.

After editing, validate each graph by booting the backend and loading them
through the existing example-loader endpoint to confirm none break.

## Files affected

### New files

| File | Role |
|---|---|
| `backend/app/nodes/control/__init__.py` | Package marker |
| `backend/app/nodes/control/start_node.py` | Start node implementation |
| `frontend/src/components/Nodes/StartNode.tsx` | Start node React component |
| `frontend/src/components/Nodes/StartNode.module.css` | Start node styles |
| `frontend/src/components/Nodes/flag-icon.svg` | Entry-point flag icon |
| `frontend/src/components/shared/MigrationModal.tsx` | "No entry points" prompt |
| `frontend/src/components/shared/MigrationModal.module.css` | Modal styles |
| `frontend/src/utils/autoLayout.ts` | Dagre integration + swim lane packing |
| `frontend/src/utils/autoLayout.test.ts` | Unit tests |

### Modified files

| File | Change |
|---|---|
| `backend/app/schemas/models.py` | Add `NodeData.isEntryPoint`, `EdgeData.type` |
| `backend/app/core/graph_engine.py` | `find_entry_points`, `reachable_from_entry_points`, gated execution, exclude trigger from in-degree |
| `backend/app/core/node_base.py` | Add `DataType.TRIGGER` enum value |
| `backend/app/core/type_system.py` | Add `DataType.TRIGGER: {DataType.TRIGGER}` to `_COMPAT` so trigger ports only connect to trigger ports (and exclude TRIGGER from `ANY`'s compatible set) |
| `backend/app/api/routes_graph.py` | Surface `noEntryPoints` validation error code so frontend can show migration modal |
| `backend/tests/test_graph_engine.py` | New test cases (see below) |
| `frontend/package.json` | `@dagrejs/dagre` dependency |
| `frontend/src/components/Nodes/BaseNode.tsx` | Apply `.entryPoint` modifier; right-click context menu |
| `frontend/src/components/Nodes/BaseNode.module.css` | `.entryPoint` styles |
| `frontend/src/components/Canvas/FlowCanvas.tsx` | Register `start` node type and `triggerEdge` edge type |
| `frontend/src/components/Sidebar/NodePalette.tsx` | Render `Control` category first |
| `frontend/src/components/Canvas/QuickNodeSearch.tsx` | Boost Start node ranking |
| `frontend/src/components/Toolbar/Toolbar.tsx` | Auto Layout split button |
| `frontend/src/store/tabStore.ts` | `applyLayout`, `undoLastLayout`, `toggleEntryPoint` actions; layout snapshot state |
| `frontend/src/store/uiStore.ts` | `lastLayoutMode` persisted setting |
| `frontend/src/hooks/useKeyboardShortcuts.ts` | `Shift+L` binding |
| `frontend/src/hooks/useGraphExecution.ts` | Surface `noEntryPoints` toast on execution attempt |
| `frontend/src/i18n/locales/en.ts` | New strings |
| `frontend/src/i18n/locales/zh-TW.ts` | New strings |
| `examples/**/graph.json` (×16) | Add `isEntryPoint: true` to root data nodes |

### Untouched (verified)

- `BaseNode` registry / auto-discovery — picks up `nodes/control/start_node.py`
  for free.
- React Flow's `Node` / `Edge` types — adding optional fields to `data` is
  type-safe via `Node<NodeData>`.
- Existing topological-sort algorithm — only the input set changes (filter to
  reachable + exclude trigger edges), the algorithm itself is unchanged.

## Testing & verification

### Backend unit tests (`backend/tests/test_graph_engine.py`)

| Case | Expected |
|---|---|
| Two disconnected components, only one has entry point | Only the entry-pointed component executes; other is silently skipped |
| Trigger edge from Start → Dataset | Dataset still treated as in-degree 0 for execution; component is gated by Start being an entry point |
| Graph with zero entry points | `validate_graph()` raises with `noEntryPoints` error code |
| Node marked `isEntryPoint=true` but has incoming data edges | `validate_graph()` raises (entry must be data-root) |
| Start → X via trigger, but X also has an incoming data edge from Y | `validate_graph()` raises (entry must be data-root, even when entry comes from a trigger edge) |
| Cycle inside an entry-pointed component | `validate_graph()` raises (current behaviour preserved) |
| Cycle inside a draft (no entry point) component | Validation passes; cycle component is skipped |
| Two entry points in the same component | Both treated as entry points, no double-execution (the component runs once via topo sort) |
| Start node connected to a node, no other entries | Component runs, Start node's `execute()` returns empty dict, downstream runs normally |

### Frontend unit tests (`frontend/src/utils/autoLayout.test.ts`)

| Case | Expected |
|---|---|
| Single linear chain `A → B → C → D` | All nodes share same Y; X strictly increasing; spacing ≈ RANKSEP |
| Diamond `A → B, A → C, B → D, C → D` | A and D centred; B and C stacked vertically at the same X |
| Two disconnected components | Components in distinct Y bands separated by LANE_GAP |
| Component with cycle `A → B → C → A` | Function returns valid coordinates (no infinite loop / NaN); back-edge handled by Dagre |
| `mode='selected'` with 3 of 6 nodes selected | Only selected nodes change position; centroid preserved |
| `mode='experiments'` with one entry-pointed and one draft component | Only entry-pointed component is repositioned |

### Manual Chrome verification (REQUIRED before marking implementation done)

The user has explicitly required browser-based verification using Chrome.
After implementation, run through this checklist using the
`mcp__claude-in-chrome__*` tooling:

1. **Boot dev environment** — start backend (`uv run uvicorn ...`) and frontend
   (`pnpm dev`), open the app in Chrome.
2. **Create entry point via marker:**
   - Drag a Dataset node onto the canvas.
   - Right-click → confirm `Set as Entry Point` appears.
   - Click it; verify the node gets the green outline + flag badge.
3. **Create entry point via Start node:**
   - Open NodePalette; verify `Control > Start` is visible at the top.
   - Drag a Start node onto the canvas.
   - Verify it has the green outline (always-on entry).
   - Drag a connection from Start's trigger output to a fresh Dataset node;
     verify the edge renders as green dashed.
4. **Execute experiment vs. draft:**
   - Build two disconnected chains (e.g. `Dataset → DataLoader → ...` ×2).
   - Mark only one as entry point.
   - Click Run; verify only the entry-pointed chain executes (check logs panel).
   - Mark the other as well; verify both run.
5. **No-entry-point validation:**
   - Remove all entry points.
   - Click Run; verify a toast appears with the "No entry points" message
     and execution does **not** start.
6. **Auto Layout — Experiments mode:**
   - Build two messy chains, one with entry point, one draft.
   - Click `Auto Layout`; verify only the entry-pointed chain rearranges
     into a clean swim lane.
7. **Auto Layout — All mode:**
   - From the Auto Layout dropdown, pick `Layout All`.
   - Verify both chains rearrange into stacked swim lanes (entry-pointed
     chain on top).
8. **Auto Layout — Selected mode:**
   - Select 3 nodes from a chain.
   - From the dropdown, pick `Layout Selected (3)`.
   - Verify only those 3 nodes move; the rest stay put.
9. **Auto Layout with cycle:**
   - Create a temporary cycle (e.g. drag an edge from a node back to its
     ancestor).
   - Click `Auto Layout`; verify it does **not** crash, the nodes are
     positioned, and the back-edge renders as a curved line going backward.
10. **Migration modal:**
    - Load `examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json` *before*
      it's been migrated (use git stash if needed).
    - Verify the migration modal appears.
    - Click `Auto-mark all root nodes`; verify the Dataset node now has the
      entry-point outline; verify Run works.
11. **Migrated example graphs:**
    - Load each migrated example via the example loader.
    - Verify no migration modal appears (they already have entry points).
    - Run `examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json`
      end-to-end and verify training progresses for at least one epoch.
    - Spot-check 2 Model_Architecture examples (e.g. ResNet, ViT) by running
      them and verifying execution starts without validation errors.
12. **Keyboard shortcut:**
    - Press `Shift+L`; verify it triggers the last-used layout mode.
13. **Undo:**
    - Run Auto Layout; press `Ctrl+Z`; verify positions revert.

Capture a GIF using `mcp__claude-in-chrome__gif_creator` of step 6 (Auto
Layout in action) so the user has a visual record of the feature working.

## Out of scope

These are explicitly **not** part of this spec — flag any pressure to expand
into them and push back to a future spec:

- **Backend cycle execution** (RNN unrolling, GAN feedback loops, etc.) — the
  semantic question of "should the engine ever support cycles in execution" is
  a separate, much larger design.
- **Animated layout transitions** — nodes snap to new positions instantly;
  smooth tween is a polish item for later.
- **Full undo/redo system** — this spec adds a single-step "undo last layout"
  only. A general history stack across all canvas mutations is its own spec.
- **Cross-tab layout synchronisation** — Auto Layout only affects the current
  tab.
- **User-configurable layout parameters** (nodesep, ranksep, lane gap) — the
  defaults are hard-coded.
- **Sub-graph internal layout** — only the top-level canvas is laid out;
  SubgraphEditorModal contents are untouched.
- **Trigger-edge port type checking on the backend** — frontend prevents
  trigger→data and data→trigger drag connections via React Flow handle types;
  backend trusts this. Adding backend-side validation is a hardening
  follow-up.

## Implementation notes

- **Flag icon source** — use a hand-written 16×16 inline SVG (no external icon
  library). The flag is a single `<path>` element styled via `currentColor` so
  it picks up the `.entryPoint` outline colour. Living in
  `frontend/src/components/Nodes/flag-icon.svg` as a static asset that both
  `BaseNode.module.css` (`background-image: url(...)`) and `StartNode.tsx`
  (imported as a React component via Vite's SVGR) consume.
- **Context menu component** — the frontend has no reusable context-menu
  primitive yet (verified during brainstorming exploration). Introduce a
  minimal one in `frontend/src/components/shared/ContextMenu.tsx` (absolutely
  positioned div anchored to cursor, click-outside dismissal, single-level
  items only). Keep it simple — this spec only needs one menu item; the
  primitive can grow as future features need it.
- **Dagre node dimensions** — use React Flow's measured `node.width` and
  `node.height` (these are set after the first render via the
  `useNodesInitialized` hook). Fall back to constants `NODE_W = 200`,
  `NODE_H = 80` only when measurements are not yet available (cold-start case).
  The Auto Layout button is disabled (greyed out) until `useNodesInitialized`
  returns `true`, eliminating the cold-start case in practice.
