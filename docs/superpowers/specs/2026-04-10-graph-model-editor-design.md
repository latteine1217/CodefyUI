# Graph-Based Model Editor — Design Spec

**Date:** 2026-04-10
**Status:** Approved for implementation
**Scope:** Replace the linear-only `SequentialModel` with a DAG-based model editor that supports skip connections, branching, and multi-input / multi-output models. Rewrite all existing examples to use the new format.

## 1. Problem

Today the `SequentialModel` node only supports linear chains:

- Backend: `SequentialModelNode.execute` parses a flat JSON layer array and builds `nn.Sequential(*modules)`. No way to express branching.
- Frontend: `SubgraphEditorModal` derives connection order from Y position via `rebuildChain`, overwriting any manual edges. There is no concept of inputs, outputs, or merge points.

This makes it impossible to express ResNet-style skip connections, UNet encoder/decoder, or any model with multiple inputs / outputs. The example named `ResNet-SkipConnection-CNN` is misleading — it currently has no real skip connections, just a linear chain.

There is no published version of CodefyUI yet, so backward compatibility with the old flat-array format is **explicitly out of scope**.

## 2. Goals

- Build any DAG of layers, not just linear chains
- Explicit `Input` / `Output` boundary nodes with multiple named ports each
- Merge layers (`Add`, `Concat`, `Multiply`, `Subtract`, `Mean`, `Stack`) for skip connections and branch combinations
- Real edges as the source of truth for topology (no Y-position chaining)
- All 16 existing examples migrated to the new format and verified to run end-to-end in Chrome
- New `UNet-Segmentation-CNN` example as a showcase

## 3. Non-Goals (v1)

- Backward compatibility with the old `[{type:...}]` flat-array format. Format change is hard-cut.
- Decomposing Transformer / LSTM / GRU wrapper blocks into their primitive constituents (LayerNorm + MHA + Add + ...). Wrapper blocks remain opaque single-input single-output layers in v1. (This is "Option Z" — deferred to a future iteration.)
- TrainingLoop / Inference node support for multi-input or multi-output models. The built `nn.Module` *can* have multi I/O, but downstream training pipelines stay single-I/O. Multi-I/O models built in v1 are usable through manual wiring or future dedicated nodes.
- Auto-layout / graph beautification. Node placement is fully manual.

## 4. Data Model — DAG JSON Format (v2)

`SequentialModel.params.layers` continues to be a JSON string parameter, but the schema becomes:

```json
{
  "version": 2,
  "nodes": [
    {
      "id": "<stable uuid>",
      "type": "<layer type>",
      "params": { ... layer constructor kwargs ... },
      "ports": [{"id": "<uuid>", "name": "x"}]
    }
  ],
  "edges": [
    {
      "id": "<uuid>",
      "source": "<node id>",
      "sourceHandle": "<port id or null>",
      "target": "<node id>",
      "targetHandle": "<port id or null>"
    }
  ]
}
```

Notes:

- Node `id`s are stable UUIDs assigned when the node is created. They survive renames.
- `Input` / `Output` nodes carry a `ports` list. Other nodes never use `ports` (they have a single implicit input handle and a single implicit output handle).
- Merge nodes (`Add`, `Concat`, `Multiply`, `Subtract`, `Mean`, `Stack`) accept an unlimited number of incoming edges to their single input handle. They produce a single output.
- `sourceHandle` / `targetHandle` are port `id`s for `Input` / `Output` nodes; `null` for plain layer nodes.
- Edge `id` is for React Flow bookkeeping; not used by the backend.

### 4.1 Validation Rules (enforced before "Apply")

- Exactly one `Input` node and one `Output` node
- `Input` has at least one port; `Output` has at least one port
- All port names within an `Input` node are unique; same for `Output`
- Each `Output` port has exactly one incoming edge
- Each `Input` port has at least one outgoing edge (no dead ports)
- Graph is acyclic
- Every node is reachable from at least one Input port (no orphans)
- Every node can reach at least one Output port (no dangling subgraphs)
- Plain (non-merge) layer nodes have at most one incoming edge

Validation failures produce a toast with a specific error message and block the Apply action.

## 5. Backend — DAG Model Builder

### 5.1 New module: `backend/app/nodes/utility/graph_model.py`

Defines:

- `GraphModelModule(nn.Module)` — the runtime model class
- `build_graph_model(spec: dict) -> GraphModelModule` — entry point
- Merge layer wrappers: `_AddLayer`, `_ConcatLayer`, `_MultiplyLayer`, `_SubtractLayer`, `_MeanLayer`, `_StackLayer`

`SequentialModelNode.execute` is rewritten to:

1. Parse `params.layers` as JSON
2. Validate `version == 2`
3. Call `build_graph_model(spec)`
4. Return `{"model": built_module}`

### 5.2 `GraphModelModule.forward`

Stores at construction time:

- `self.layers: nn.ModuleDict` — key is node id, value is the wrapped `nn.Module` (skipped for `Input` / `Output` markers)
- `self.topo_order: list[str]` — topological sort of node ids
- `self.input_ports: list[tuple[str, str]]` — list of `(node_id, port_name)` for the Input node, in port order
- `self.output_ports: list[tuple[str, str]]` — same for the Output node
- `self.incoming: dict[str, list[tuple[str, str | None]]]` — node id → list of (source_id, source_handle), in declaration order
- `self.input_node_id: str`, `self.output_node_id: str`

`forward(*args, **kwargs)`:

1. Resolve named inputs:
   - If only `args` are provided, zip with `self.input_ports` by index
   - If only `kwargs`, look them up by port name
   - Mixed is an error
   - Single Input port + single positional arg: standard `model(tensor)` shorthand
2. Initialize `values: dict[str, Tensor]` keyed by `(node_id, port_name | None)`. Seed with the resolved inputs at `(input_node_id, port_name)`.
3. Iterate `topo_order`:
   - `Input` node: skip (already seeded)
   - `Output` node: collect `values[(src, src_handle)]` for each incoming edge, keyed by the *output port name*
   - Merge node: gather all incoming tensors, apply the corresponding op, store at `(node_id, None)`
   - Plain layer: take the single incoming tensor, run `self.layers[node_id]`, store at `(node_id, None)`
4. Build the result dict from collected outputs
5. **Single-output shorthand:** if there is exactly one Output port, return the tensor directly instead of a dict. This keeps `model(x) -> tensor` working for the common case so TrainingLoop / Inference are unchanged.

### 5.3 Layer Builders

Existing `_build_layer` is reused for plain layers. New entries are added for merge layers via the wrappers above. `Input` and `Output` are not built as nn.Modules; they are recognized by name and handled specially in `build_graph_model`.

Merge layer constructors:

- `Add()`, `Multiply()` — no params, accept 2+ inputs, applied pairwise left-to-right
- `Subtract()` — no params, **requires exactly 2 inputs** (computes `a - b` in incoming-edge order); validation error otherwise
- `Mean()` — no params, element-wise mean across all incoming tensors
- `Concat(dim: int = 1)`, `Stack(dim: int = 1)` — `dim` param; accept 2+ inputs

### 5.4 Validation in the Backend

`build_graph_model` re-runs the same validation rules as the frontend, raising `ValueError` with the exact same messages on failure. This protects against hand-edited example JSONs.

## 6. Frontend — DAG Editor

### 6.1 Files Touched / Added

- **Edited:** `frontend/src/components/SubgraphEditor/SubgraphEditorModal.tsx`
  - Remove `rebuildChain` and all Y-position auto-chain logic
  - Add validation pass on Apply
  - Add edge connection handling (free-form, not auto-rebuilt)
  - Add new layer palette categories: `I/O`, `Merge`
- **Edited:** `frontend/src/components/SubgraphEditor/LayerNode.tsx`
  - Plain layer nodes: enforce single incoming edge by limiting `targetHandle` connectability
  - Merge layer nodes: allow multiple incoming edges (no limit)
- **New:** `frontend/src/components/SubgraphEditor/InputNode.tsx`
  - Custom React Flow node for the `Input` boundary
  - Renders one labeled output handle per port along the bottom edge
  - No input handle
- **New:** `frontend/src/components/SubgraphEditor/OutputNode.tsx`
  - Custom React Flow node for the `Output` boundary
  - Renders one labeled input handle per port along the top edge
  - No output handle
- **New:** `frontend/src/components/SubgraphEditor/PortListEditor.tsx`
  - Param-editor panel UI shown when an `Input` or `Output` node is selected
  - Add / remove / rename / reorder ports
  - Removing a port also removes all edges that reference it
  - Validates duplicate names inline (red border + tooltip)
- **Edited:** `frontend/src/i18n/locales/en.ts` and `zh-TW.ts`
  - New strings for I/O / Merge categories, validation errors, port management UI

### 6.2 Editor Behavior

**Initial state of a fresh model:** Pre-populated with one `Input` node (one port named `x`) and one `Output` node (one port named `out`), positioned at top-left and bottom-right respectively. Empty in between.

**Adding a layer:** Drag from palette as today. Plain layers get an input handle on top and an output handle on bottom. Merge layers look the same but accept multiple inbound edges.

**Connecting:** React Flow's standard connection flow. Hover near a handle, drag to another. Validity is enforced via `isValidConnection` callback:

- Plain layer's input handle: rejects if it already has an incoming edge
- Output node port: rejects if it already has an incoming edge
- All other connections: allowed

**Selecting an `Input` / `Output` node:** Right-side param panel shows the `PortListEditor` with the port list, instead of the standard layer param editor.

**Apply:** Runs the validation rules from §4.1. On any failure, shows a toast with the specific error and does not close the modal. On success, serializes to v2 JSON and writes back to the parent node's `layers` param.

### 6.3 Serialization

Frontend stores nodes/edges in React Flow format internally, then serializes to the v2 schema on Apply. On Open, deserializes the v2 schema into React Flow state.

Conversion functions live alongside the modal:

- `graphToFlow(json: string): { nodes, edges }`
- `flowToGraphJson(nodes, edges): string`

These replace the existing `layersToFlow` / `flowToLayersJson`.

## 7. Examples — Rewrite Plan

All 16 existing examples use the old format. Each is rewritten by hand (no script) and verified in Chrome. Order:

1. **TrainCNN-MNIST** (Usage_Example) — linear, end-to-end runnable. First because it's the simplest end-to-end smoke test for the whole pipeline.
2. **InferenceCNN-MNIST** (Usage_Example) — linear, runs against the saved weights from #1.
3. **ResNet-SkipConnection-CNN** (Model_Architecture) — rewritten with **real residual blocks** using `Add` merge nodes. This is the showcase for skip connections. 2-3 residual blocks, each: `Conv → BN → ReLU → Conv → BN → Add(skip) → ReLU`.
4. **NEW: UNet-Segmentation-CNN** (Model_Architecture) — new file. Standard 4-level UNet with `Concat`-based encoder-to-decoder skip connections.
5. **ConvNeXt-CNN, EfficientNet-CNN** — linear hand-rewrites.
6. **ViT-ImageClassifier-Transformer, BERT-Encoder-Transformer, GPT-DecoderOnly-Transformer, LLaMA-Decoder-Transformer, SwinTransformer-Transformer, DiT-Diffusion-Transformer** — linear hand-rewrites using the existing wrapper blocks (Option Y). Internal residual structure is encapsulated inside the wrappers in v1. Future Option Z will decompose them.
7. **Seq2Seq-Attention-RNN, BiGRU-SpeechRecognition-RNN, TimeSeries-LSTM-RNN** — linear hand-rewrites using LSTM/GRU/MultiHeadAttention wrapper blocks.
8. **DQN-Atari-RL, PPO-Robotics-RL** — linear hand-rewrites.

### 7.1 Verification per Example

For each example:

1. Reload backend (auto via FastAPI hot reload) and frontend (Vite HMR)
2. Open the example from the Empty Canvas overlay or Examples menu
3. Double-click the SequentialModel node to open the editor — verify the DAG renders correctly
4. Close editor, click "Run Pipeline"
5. **Pure Model_Architecture examples** (no Dataset / TrainingLoop): verify the model builds, parameter count is reasonable, no errors. Add a "PrintModel" or use the existing Print node to surface the model summary.
6. **End-to-end examples** (TrainCNN, InferenceCNN, ResNet, UNet if extended): run at least 1 batch / 1 forward pass without errors

If any verification fails, fix and re-verify before moving to the next example.

## 8. Testing

**Backend unit tests** — new file `backend/tests/test_graph_model.py`:

- `test_linear_dag` — Input → Linear → Output, equivalent to `nn.Linear` alone
- `test_residual_block` — Input → Conv → ReLU → Add(skip from Input) → Output, output matches hand-written nn.Module
- `test_unet_skip` — small encoder-decoder with one Concat skip, shape check
- `test_multi_input` — two Input ports feeding an Add, called via `model(x=t1, y=t2)`
- `test_multi_output` — one Input, two Output ports, returns dict with both keys
- `test_single_io_shorthand` — single Input, single Output: `model(x)` returns a tensor (not a dict)
- `test_validation_cycle` — DAG with a cycle raises ValueError
- `test_validation_no_input` — missing Input raises
- `test_validation_no_output` — missing Output raises
- `test_validation_duplicate_port_names` — two ports with the same name raises
- `test_validation_unconnected_output_port` — Output port with zero incoming edges raises
- `test_validation_multiple_edges_to_plain_layer` — plain layer with two inputs raises (must use a Merge layer)

**Frontend smoke test** — manual via Chrome browser tooling:

- Build a residual block from scratch
- Verify validation toast on cycle
- Verify removing an Input port removes its edges
- Verify renaming an Input port survives across editor open/close
- Verify each rewritten example loads, edits, saves, and runs

## 9. Error Handling

Validation errors are raised as `ValueError` from the backend with explicit messages, surfaced to the frontend through the existing graph execution error path (toast + execution log entry).

Frontend validation errors (during Apply) use the existing `useToastStore.addToast(msg, 'error')` mechanism. Same for port-management errors in the PortListEditor (e.g., duplicate name).

## 10. Migration & Rollout

Single-shot replacement. No feature flag, no gradual rollout. Order of operations:

1. Land backend `graph_model.py` + rewritten `SequentialModelNode.execute`
2. Land frontend editor changes (custom Input/Output nodes, port editor, validation, free-form edges)
3. Run backend tests
4. Rewrite examples one at a time per §7, verifying each in Chrome before moving on
5. Commit each rewritten example as its own commit (so failures are easy to bisect)

## 11. Future Work (out of scope)

- **Option Z**: Decompose Transformer / LSTM / GRU wrapper blocks into their primitive operations so the residual structure inside them is visible and editable
- **Multi-I/O downstream support**: Adapt TrainingLoop and Inference nodes (or add dedicated MultiTrainingLoop / MultiInference variants) to handle multi-input / multi-output models
- **Auto-layout**: Optional one-click "tidy up" that runs a graph layout algorithm (Dagre / ELK) to arrange nodes
- **Subgraph reuse**: Allow saving a subgraph as a reusable composite layer that can be dragged into another model
