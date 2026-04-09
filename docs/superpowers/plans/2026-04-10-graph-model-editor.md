# Graph-Based Model Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the linear-only `SequentialModel` with a DAG-based model editor that supports skip connections, branching, and multi-input/multi-output models. Rewrite all 16 existing examples to use the new format and verify each works in Chrome.

**Architecture:** Backend introduces `GraphModelModule(nn.Module)` whose `forward` walks a topologically-sorted DAG. `SequentialModelNode.execute` is rewritten to parse a v2 JSON schema (`{version, nodes, edges}`) and call `build_graph_model`. Frontend `SubgraphEditorModal` is rewritten: removes Y-position auto-chaining, adds custom `InputNode`/`OutputNode` components with dynamic per-port handles, adds Merge layer category, and validates DAG structure on Apply. All 16 examples are hand-rewritten with the new format and verified end-to-end in Chrome one by one.

**Tech Stack:** Python 3.11+, PyTorch (`torch.nn`), pytest, FastAPI; React 18, TypeScript, `@xyflow/react` (React Flow), Zustand, Vite.

**Spec:** `docs/superpowers/specs/2026-04-10-graph-model-editor-design.md`

---

## File Structure

**Backend (new + modified):**
- `backend/app/nodes/utility/graph_model.py` (new) — `GraphModelModule`, `build_graph_model`, merge layer wrappers, validation
- `backend/app/nodes/utility/sequential_node.py` (rewrite `execute`, drop old `_build_layer` flat-array path) — keep `_build_layer` and complex wrappers but invoke from `graph_model`
- `backend/tests/test_graph_model.py` (new) — unit tests for the DAG builder

**Frontend (new + modified):**
- `frontend/src/components/SubgraphEditor/SubgraphEditorModal.tsx` (rewrite editor body — remove `rebuildChain`, free-form edges, add validation, swap serialization, add Merge palette section)
- `frontend/src/components/SubgraphEditor/LayerNode.tsx` (modify — accept `isMerge` flag for handle behavior)
- `frontend/src/components/SubgraphEditor/InputNode.tsx` (new) — dynamic output handles
- `frontend/src/components/SubgraphEditor/OutputNode.tsx` (new) — dynamic input handles
- `frontend/src/components/SubgraphEditor/PortListEditor.tsx` (new) — port CRUD UI
- `frontend/src/components/SubgraphEditor/graphSerialization.ts` (new) — `graphToFlow`, `flowToGraphJson`, validation
- `frontend/src/i18n/locales/en.ts` (modify — add I/O / Merge / validation strings)
- `frontend/src/i18n/locales/zh-TW.ts` (modify — same)

**Examples (rewritten — 16 files + 1 new):**
- `examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json`
- `examples/Usage_Example/CNN-MNIST/InferenceCNN-MNIST/graph.json`
- `examples/Model_Architecture/ResNet-SkipConnection-CNN/graph.json` (real residual blocks)
- `examples/Model_Architecture/UNet-Segmentation-CNN/graph.json` (NEW)
- `examples/Model_Architecture/ConvNeXt-CNN/graph.json`
- `examples/Model_Architecture/EfficientNet-CNN/graph.json`
- `examples/Model_Architecture/ViT-ImageClassifier-Transformer/graph.json`
- `examples/Model_Architecture/BERT-Encoder-Transformer/graph.json`
- `examples/Model_Architecture/GPT-DecoderOnly-Transformer/graph.json`
- `examples/Model_Architecture/LLaMA-Decoder-Transformer/graph.json`
- `examples/Model_Architecture/SwinTransformer-Transformer/graph.json`
- `examples/Model_Architecture/DiT-Diffusion-Transformer/graph.json`
- `examples/Model_Architecture/Seq2Seq-Attention-RNN/graph.json`
- `examples/Model_Architecture/BiGRU-SpeechRecognition-RNN/graph.json`
- `examples/Model_Architecture/TimeSeries-LSTM-RNN/graph.json`
- `examples/Model_Architecture/DQN-Atari-RL/graph.json`
- `examples/Model_Architecture/PPO-Robotics-RL/graph.json`

---

## Phase 1 — Backend DAG Model

### Task 1: Skeleton GraphModelModule + first failing test

**Files:**
- Create: `backend/app/nodes/utility/graph_model.py`
- Create: `backend/tests/test_graph_model.py`

- [ ] **Step 1: Write the failing test for a single Linear layer DAG**

```python
# backend/tests/test_graph_model.py
import json

import pytest
import torch

from app.nodes.utility.graph_model import build_graph_model


def _spec(nodes, edges):
    return {"version": 2, "nodes": nodes, "edges": edges}


def test_linear_dag_single_layer():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_in", "name": "x"}]},
            {"id": "lin", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_out", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_in", "target": "lin", "targetHandle": None},
            {"id": "e2", "source": "lin", "sourceHandle": None, "target": "out", "targetHandle": "p_out"},
        ],
    )

    model = build_graph_model(spec)
    x = torch.randn(3, 4)
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_graph_model.py::test_linear_dag_single_layer -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.nodes.utility.graph_model'`

- [ ] **Step 3: Create graph_model.py with minimal `build_graph_model` to pass the test**

```python
# backend/app/nodes/utility/graph_model.py
"""DAG-based model builder for the SequentialModel node.

Builds an nn.Module whose forward pass walks a topologically-sorted DAG of
layer nodes. Supports explicit Input/Output boundary nodes with named ports,
plain layers, and merge layers (Add, Concat, Multiply, Subtract, Mean, Stack).
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .sequential_node import _build_layer  # reuse existing builders

logger = logging.getLogger(__name__)


_MERGE_TYPES = {"Add", "Concat", "Multiply", "Subtract", "Mean", "Stack"}
_BOUNDARY_TYPES = {"Input", "Output"}


class GraphModelModule(nn.Module):
    """nn.Module that executes a DAG of layers in topological order."""

    def __init__(
        self,
        layers: dict[str, nn.Module],
        merge_specs: dict[str, dict[str, Any]],
        topo_order: list[str],
        incoming: dict[str, list[tuple[str, str | None]]],
        input_node_id: str,
        output_node_id: str,
        input_ports: list[tuple[str, str]],
        output_ports: list[tuple[str, str]],
    ):
        super().__init__()
        self.layers = nn.ModuleDict(layers)
        self.merge_specs = merge_specs
        self.topo_order = topo_order
        self.incoming = incoming
        self.input_node_id = input_node_id
        self.output_node_id = output_node_id
        self.input_ports = input_ports
        self.output_ports = output_ports

    def forward(self, *args, **kwargs):
        # Resolve named inputs
        if args and kwargs:
            raise ValueError("GraphModel.forward: pass either positional or keyword args, not both")
        if args:
            if len(args) != len(self.input_ports):
                raise ValueError(
                    f"GraphModel.forward: expected {len(self.input_ports)} positional args, got {len(args)}"
                )
            named = {port_name: arg for (_, port_name), arg in zip(self.input_ports, args)}
        else:
            for _, port_name in self.input_ports:
                if port_name not in kwargs:
                    raise ValueError(f"GraphModel.forward: missing required input '{port_name}'")
            named = {port_name: kwargs[port_name] for _, port_name in self.input_ports}

        values: dict[tuple[str, str | None], torch.Tensor] = {}
        for _, port_name in self.input_ports:
            values[(self.input_node_id, port_name)] = named[port_name]

        outputs: dict[str, torch.Tensor] = {}
        for node_id in self.topo_order:
            if node_id == self.input_node_id:
                continue
            if node_id == self.output_node_id:
                # Each output port: collect single incoming tensor
                for port_id, port_name in self.output_ports:
                    incoming_edges = [
                        (src, sh) for (src, sh, target_handle) in self.incoming.get(node_id, [])
                        if target_handle == port_id
                    ]
                    if len(incoming_edges) != 1:
                        raise ValueError(
                            f"Output port '{port_name}' must have exactly one incoming edge, got {len(incoming_edges)}"
                        )
                    src, sh = incoming_edges[0]
                    outputs[port_name] = values[(src, sh)]
                continue

            inputs = [values[(src, sh)] for (src, sh, _) in self.incoming.get(node_id, [])]

            if node_id in self.merge_specs:
                spec = self.merge_specs[node_id]
                values[(node_id, None)] = _apply_merge(spec["type"], inputs, spec.get("params", {}))
            else:
                if len(inputs) != 1:
                    raise ValueError(f"Plain layer '{node_id}' expects 1 input, got {len(inputs)}")
                values[(node_id, None)] = self.layers[node_id](inputs[0])

        if len(self.output_ports) == 1:
            return outputs[self.output_ports[0][1]]
        return outputs


def _apply_merge(merge_type: str, inputs: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
    if merge_type == "Add":
        if len(inputs) < 2:
            raise ValueError("Add requires 2+ inputs")
        out = inputs[0]
        for t in inputs[1:]:
            out = out + t
        return out
    if merge_type == "Multiply":
        if len(inputs) < 2:
            raise ValueError("Multiply requires 2+ inputs")
        out = inputs[0]
        for t in inputs[1:]:
            out = out * t
        return out
    if merge_type == "Subtract":
        if len(inputs) != 2:
            raise ValueError("Subtract requires exactly 2 inputs")
        return inputs[0] - inputs[1]
    if merge_type == "Mean":
        if len(inputs) < 2:
            raise ValueError("Mean requires 2+ inputs")
        return torch.stack(inputs, dim=0).mean(dim=0)
    if merge_type == "Concat":
        dim = int(params.get("dim", 1))
        return torch.cat(inputs, dim=dim)
    if merge_type == "Stack":
        dim = int(params.get("dim", 1))
        return torch.stack(inputs, dim=dim)
    raise ValueError(f"Unknown merge type: {merge_type}")


def build_graph_model(spec: dict[str, Any]) -> GraphModelModule:
    """Build a GraphModelModule from a v2 spec dict."""
    if spec.get("version") != 2:
        raise ValueError(f"Unsupported graph spec version: {spec.get('version')}")

    nodes = spec.get("nodes", [])
    edges = spec.get("edges", [])

    nodes_by_id = {n["id"]: n for n in nodes}

    input_nodes = [n for n in nodes if n["type"] == "Input"]
    output_nodes = [n for n in nodes if n["type"] == "Output"]
    if len(input_nodes) != 1:
        raise ValueError(f"Graph must have exactly one Input node, found {len(input_nodes)}")
    if len(output_nodes) != 1:
        raise ValueError(f"Graph must have exactly one Output node, found {len(output_nodes)}")

    input_node = input_nodes[0]
    output_node = output_nodes[0]

    input_ports_raw = input_node.get("ports", [])
    output_ports_raw = output_node.get("ports", [])
    if not input_ports_raw:
        raise ValueError("Input node must have at least one port")
    if not output_ports_raw:
        raise ValueError("Output node must have at least one port")

    in_names = [p["name"] for p in input_ports_raw]
    if len(set(in_names)) != len(in_names):
        raise ValueError("Input port names must be unique")
    out_names = [p["name"] for p in output_ports_raw]
    if len(set(out_names)) != len(out_names):
        raise ValueError("Output port names must be unique")

    input_ports = [(p["id"], p["name"]) for p in input_ports_raw]
    output_ports = [(p["id"], p["name"]) for p in output_ports_raw]

    # incoming[node_id] = list of (src_id, src_handle, target_handle)
    incoming: dict[str, list[tuple[str, str | None, str | None]]] = {n["id"]: [] for n in nodes}
    outgoing: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        src, tgt = e["source"], e["target"]
        if src not in nodes_by_id or tgt not in nodes_by_id:
            raise ValueError(f"Edge references unknown node: {src} -> {tgt}")
        incoming[tgt].append((src, e.get("sourceHandle"), e.get("targetHandle")))
        outgoing[src].append(tgt)

    # Build layers + merge specs
    layers: dict[str, nn.Module] = {}
    merge_specs: dict[str, dict[str, Any]] = {}
    for n in nodes:
        t = n["type"]
        if t in _BOUNDARY_TYPES:
            continue
        if t in _MERGE_TYPES:
            merge_specs[n["id"]] = {"type": t, "params": n.get("params", {})}
            continue
        cfg = {"type": t, **n.get("params", {})}
        layers[n["id"]] = _build_layer(cfg)

    # Topological sort (Kahn's algorithm)
    in_degree = {nid: len(incoming[nid]) for nid in nodes_by_id}
    queue = [nid for nid, d in in_degree.items() if d == 0]
    topo_order: list[str] = []
    while queue:
        nid = queue.pop(0)
        topo_order.append(nid)
        for tgt in outgoing[nid]:
            in_degree[tgt] -= 1
            if in_degree[tgt] == 0:
                queue.append(tgt)
    if len(topo_order) != len(nodes_by_id):
        raise ValueError("Graph contains a cycle")

    # Validation: each Output port has exactly 1 incoming, each Input port used 1+
    out_port_ids = {p["id"] for p in output_ports_raw}
    in_port_ids = {p["id"] for p in input_ports_raw}
    out_port_use: dict[str, int] = {pid: 0 for pid in out_port_ids}
    in_port_use: dict[str, int] = {pid: 0 for pid in in_port_ids}
    for e in edges:
        if e["target"] == output_node["id"]:
            th = e.get("targetHandle")
            if th not in out_port_ids:
                raise ValueError(f"Edge into Output references unknown port id: {th}")
            out_port_use[th] += 1
        if e["source"] == input_node["id"]:
            sh = e.get("sourceHandle")
            if sh not in in_port_ids:
                raise ValueError(f"Edge from Input references unknown port id: {sh}")
            in_port_use[sh] += 1
    for pid, count in out_port_use.items():
        if count != 1:
            name = next(p["name"] for p in output_ports_raw if p["id"] == pid)
            raise ValueError(f"Output port '{name}' must have exactly one incoming edge, got {count}")
    for pid, count in in_port_use.items():
        if count < 1:
            name = next(p["name"] for p in input_ports_raw if p["id"] == pid)
            raise ValueError(f"Input port '{name}' is unused")

    # Plain layers must have exactly 1 incoming
    for nid, edges_in in incoming.items():
        n = nodes_by_id[nid]
        if n["type"] in _BOUNDARY_TYPES or n["type"] in _MERGE_TYPES:
            continue
        if len(edges_in) != 1:
            raise ValueError(f"Plain layer '{nid}' ({n['type']}) must have exactly 1 incoming edge, got {len(edges_in)}")

    incoming_for_module: dict[str, list[tuple[str, str | None]]] = {
        nid: [(src, sh) for (src, sh, _th) in lst] for nid, lst in incoming.items()
    }
    # For Output node, keep target_handle so forward can group by output port
    incoming_with_target = {output_node["id"]: incoming[output_node["id"]]}

    # Patch: pass full triple for output node
    class _ModuleWithOutputIncoming(GraphModelModule):
        def forward(self, *args, **kwargs):
            self.incoming_full = incoming_with_target  # type: ignore
            return super().forward(*args, **kwargs)

    model = GraphModelModule(
        layers=layers,
        merge_specs=merge_specs,
        topo_order=topo_order,
        incoming=incoming_for_module,
        input_node_id=input_node["id"],
        output_node_id=output_node["id"],
        input_ports=input_ports,
        output_ports=output_ports,
    )
    # Stash full incoming (with target handles) for the Output node only
    model._output_incoming_full = incoming[output_node["id"]]  # type: ignore[attr-defined]
    return model
```

Note: The forward method as written needs to know the target handle for Output edges. Update `forward` to read `self._output_incoming_full` for that node:

Replace the Output handling in `forward` with:

```python
            if node_id == self.output_node_id:
                full = getattr(self, "_output_incoming_full", [])
                for port_id, port_name in self.output_ports:
                    incoming_edges = [(src, sh) for (src, sh, th) in full if th == port_id]
                    if len(incoming_edges) != 1:
                        raise ValueError(
                            f"Output port '{port_name}' must have exactly one incoming edge, got {len(incoming_edges)}"
                        )
                    src, sh = incoming_edges[0]
                    outputs[port_name] = values[(src, sh)]
                continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_graph_model.py::test_linear_dag_single_layer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/nodes/utility/graph_model.py backend/tests/test_graph_model.py
git commit -m "feat(backend): add GraphModelModule with topo-sorted forward pass"
```

---

### Task 2: Residual block test (Add merge)

**Files:**
- Modify: `backend/tests/test_graph_model.py`

- [ ] **Step 1: Add the failing test**

Append to `backend/tests/test_graph_model.py`:

```python
def test_residual_block_with_add():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_in", "name": "x"}]},
            {"id": "lin1", "type": "Linear", "params": {"in_features": 4, "out_features": 4}},
            {"id": "relu", "type": "ReLU"},
            {"id": "lin2", "type": "Linear", "params": {"in_features": 4, "out_features": 4}},
            {"id": "add", "type": "Add"},
            {"id": "out", "type": "Output", "ports": [{"id": "p_out", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_in", "target": "lin1"},
            {"id": "e2", "source": "lin1", "target": "relu"},
            {"id": "e3", "source": "relu", "target": "lin2"},
            {"id": "e4", "source": "lin2", "target": "add"},
            {"id": "e5", "source": "in", "sourceHandle": "p_in", "target": "add"},  # skip
            {"id": "e6", "source": "add", "target": "out", "targetHandle": "p_out"},
        ],
    )
    model = build_graph_model(spec)
    x = torch.randn(2, 4)
    y = model(x)
    assert y.shape == (2, 4)
    # Skip means at least one parameter pathway exists; verify gradient flows
    y.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
```

- [ ] **Step 2: Run test**

Run: `cd backend && pytest tests/test_graph_model.py::test_residual_block_with_add -v`
Expected: PASS (the existing implementation already supports this case)

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_graph_model.py
git commit -m "test: residual block via Add merge layer"
```

---

### Task 3: Multi-input and multi-output tests

**Files:**
- Modify: `backend/tests/test_graph_model.py`

- [ ] **Step 1: Add the failing tests**

```python
def test_multi_input():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [
                {"id": "p_a", "name": "a"},
                {"id": "p_b", "name": "b"},
            ]},
            {"id": "add", "type": "Add"},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_a", "target": "add"},
            {"id": "e2", "source": "in", "sourceHandle": "p_b", "target": "add"},
            {"id": "e3", "source": "add", "target": "out", "targetHandle": "p_y"},
        ],
    )
    model = build_graph_model(spec)
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([10.0, 20.0])
    y = model(a=a, b=b)
    assert torch.allclose(y, torch.tensor([11.0, 22.0]))


def test_multi_output_returns_dict():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "lin", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "relu", "type": "ReLU"},
            {"id": "out", "type": "Output", "ports": [
                {"id": "p_raw", "name": "raw"},
                {"id": "p_act", "name": "activated"},
            ]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "lin"},
            {"id": "e2", "source": "lin", "target": "relu"},
            {"id": "e3", "source": "lin", "target": "out", "targetHandle": "p_raw"},
            {"id": "e4", "source": "relu", "target": "out", "targetHandle": "p_act"},
        ],
    )
    model = build_graph_model(spec)
    x = torch.randn(3, 4)
    out = model(x=x)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"raw", "activated"}
    assert out["raw"].shape == (3, 2)
    assert out["activated"].shape == (3, 2)
```

- [ ] **Step 2: Run tests**

Run: `cd backend && pytest tests/test_graph_model.py::test_multi_input tests/test_graph_model.py::test_multi_output_returns_dict -v`
Expected: PASS (current impl should handle both)

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_graph_model.py
git commit -m "test: multi-input and multi-output graph models"
```

---

### Task 4: UNet-like skip with Concat

**Files:**
- Modify: `backend/tests/test_graph_model.py`

- [ ] **Step 1: Add the failing test**

```python
def test_unet_skip_with_concat():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "down1", "type": "Conv2d", "params": {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "pool", "type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
            {"id": "deep", "type": "Conv2d", "params": {"in_channels": 4, "out_channels": 4, "kernel_size": 3, "padding": 1}},
            {"id": "up", "type": "ConvTranspose2d", "params": {"in_channels": 4, "out_channels": 4, "kernel_size": 2, "stride": 2}},
            {"id": "concat", "type": "Concat", "params": {"dim": 1}},
            {"id": "final", "type": "Conv2d", "params": {"in_channels": 8, "out_channels": 1, "kernel_size": 1}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "down1"},
            {"id": "e2", "source": "down1", "target": "pool"},
            {"id": "e3", "source": "pool", "target": "deep"},
            {"id": "e4", "source": "deep", "target": "up"},
            {"id": "e5", "source": "up", "target": "concat"},
            {"id": "e6", "source": "down1", "target": "concat"},  # skip
            {"id": "e7", "source": "concat", "target": "final"},
            {"id": "e8", "source": "final", "target": "out", "targetHandle": "p_y"},
        ],
    )
    model = build_graph_model(spec)
    x = torch.randn(2, 1, 16, 16)
    y = model(x)
    assert y.shape == (2, 1, 16, 16)
```

- [ ] **Step 2: Run test**

Run: `cd backend && pytest tests/test_graph_model.py::test_unet_skip_with_concat -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_graph_model.py
git commit -m "test: UNet-style skip connection via Concat"
```

---

### Task 5: Validation tests (cycles, missing I/O, duplicates, dead ports, multi-incoming plain layer)

**Files:**
- Modify: `backend/tests/test_graph_model.py`

- [ ] **Step 1: Add validation failure tests**

```python
def test_validation_cycle():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "a", "type": "Linear", "params": {"in_features": 4, "out_features": 4}},
            {"id": "b", "type": "Linear", "params": {"in_features": 4, "out_features": 4}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "a"},
            {"id": "e2", "source": "a", "target": "b"},
            {"id": "e3", "source": "b", "target": "a"},  # cycle
            {"id": "e4", "source": "b", "target": "out", "targetHandle": "p_y"},
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        build_graph_model(spec)


def test_validation_no_input():
    spec = _spec(
        nodes=[
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[],
    )
    with pytest.raises(ValueError, match="exactly one Input"):
        build_graph_model(spec)


def test_validation_no_output():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
        ],
        edges=[],
    )
    with pytest.raises(ValueError, match="exactly one Output"):
        build_graph_model(spec)


def test_validation_duplicate_input_port_names():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [
                {"id": "p1", "name": "x"},
                {"id": "p2", "name": "x"},
            ]},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[],
    )
    with pytest.raises(ValueError, match="Input port names must be unique"):
        build_graph_model(spec)


def test_validation_unconnected_output_port():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "lin", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "lin"},
            # no edge into output
        ],
    )
    with pytest.raises(ValueError, match="Output port 'y' must have exactly one incoming edge"):
        build_graph_model(spec)


def test_validation_plain_layer_multi_input():
    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [
                {"id": "p_a", "name": "a"},
                {"id": "p_b", "name": "b"},
            ]},
            {"id": "lin", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_a", "target": "lin"},
            {"id": "e2", "source": "in", "sourceHandle": "p_b", "target": "lin"},  # plain layer can't take 2 inputs
            {"id": "e3", "source": "lin", "target": "out", "targetHandle": "p_y"},
        ],
    )
    with pytest.raises(ValueError, match="must have exactly 1 incoming edge"):
        build_graph_model(spec)
```

- [ ] **Step 2: Run all validation tests**

Run: `cd backend && pytest tests/test_graph_model.py -v -k validation`
Expected: PASS for all 6

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_graph_model.py
git commit -m "test: graph model validation rules"
```

---

### Task 6: Wire SequentialModelNode to use the new builder

**Files:**
- Modify: `backend/app/nodes/utility/sequential_node.py`
- Modify: `backend/tests/test_graph_model.py`

- [ ] **Step 1: Write a failing integration test that goes through the node**

Append to `backend/tests/test_graph_model.py`:

```python
def test_sequential_node_executes_v2_spec():
    from app.nodes.utility.sequential_node import SequentialModelNode
    import json as _json

    spec = _spec(
        nodes=[
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "lin", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        edges=[
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "lin"},
            {"id": "e2", "source": "lin", "target": "out", "targetHandle": "p_y"},
        ],
    )
    node = SequentialModelNode()
    result = node.execute(inputs={}, params={"layers": _json.dumps(spec)})
    model = result["model"]
    y = model(torch.randn(3, 4))
    assert y.shape == (3, 2)
```

- [ ] **Step 2: Run test (expected to fail because old execute parses flat array)**

Run: `cd backend && pytest tests/test_graph_model.py::test_sequential_node_executes_v2_spec -v`
Expected: FAIL with `Unsupported graph spec version` or similar JSON parsing error

- [ ] **Step 3: Rewrite `SequentialModelNode.execute`**

Replace the existing `execute` method in `backend/app/nodes/utility/sequential_node.py` (around line 268-282) with:

```python
    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import json
        from .graph_model import build_graph_model

        layers_str = params.get("layers", "{}")
        spec = json.loads(layers_str)
        model = build_graph_model(spec)

        total = sum(p.numel() for p in model.parameters())
        layer_count = len(model.layers)
        logger.info("Built graph model with %d layers, %s parameters", layer_count, f"{total:,}")

        return {"model": model}
```

Also update the default value of `layers` param (around line 252-264) to a v2 spec:

```python
                default=(
                    '{"version":2,'
                    '"nodes":['
                    '{"id":"in","type":"Input","ports":[{"id":"p_x","name":"x"}]},'
                    '{"id":"c1","type":"Conv2d","params":{"in_channels":1,"out_channels":32,"kernel_size":3,"padding":1}},'
                    '{"id":"r1","type":"ReLU"},'
                    '{"id":"p1","type":"MaxPool2d","params":{"kernel_size":2,"stride":2}},'
                    '{"id":"c2","type":"Conv2d","params":{"in_channels":32,"out_channels":64,"kernel_size":3,"padding":1}},'
                    '{"id":"r2","type":"ReLU"},'
                    '{"id":"p2","type":"MaxPool2d","params":{"kernel_size":2,"stride":2}},'
                    '{"id":"f","type":"Flatten"},'
                    '{"id":"l1","type":"Linear","params":{"in_features":3136,"out_features":128}},'
                    '{"id":"r3","type":"ReLU"},'
                    '{"id":"l2","type":"Linear","params":{"in_features":128,"out_features":10}},'
                    '{"id":"out","type":"Output","ports":[{"id":"p_y","name":"y"}]}'
                    '],'
                    '"edges":['
                    '{"id":"e1","source":"in","sourceHandle":"p_x","target":"c1"},'
                    '{"id":"e2","source":"c1","target":"r1"},'
                    '{"id":"e3","source":"r1","target":"p1"},'
                    '{"id":"e4","source":"p1","target":"c2"},'
                    '{"id":"e5","source":"c2","target":"r2"},'
                    '{"id":"e6","source":"r2","target":"p2"},'
                    '{"id":"e7","source":"p2","target":"f"},'
                    '{"id":"e8","source":"f","target":"l1"},'
                    '{"id":"e9","source":"l1","target":"r3"},'
                    '{"id":"e10","source":"r3","target":"l2"},'
                    '{"id":"e11","source":"l2","target":"out","targetHandle":"p_y"}'
                    ']}'
                ),
                description="DAG JSON (v2 schema) describing the model graph.",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_graph_model.py::test_sequential_node_executes_v2_spec -v`
Expected: PASS

- [ ] **Step 5: Run the full backend test suite to confirm nothing else broke**

Run: `cd backend && pytest tests/ -x --ignore=tests/test_graph_model.py`
Expected: PASS (any failures here mean another test was relying on flat-array format — fix or update those)

Then run everything:
Run: `cd backend && pytest tests/ -x`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/nodes/utility/sequential_node.py backend/tests/test_graph_model.py
git commit -m "feat(backend): rewrite SequentialModelNode to use v2 DAG spec"
```

---

## Phase 2 — Frontend Editor

### Task 7: Add graphSerialization helpers + i18n strings

**Files:**
- Create: `frontend/src/components/SubgraphEditor/graphSerialization.ts`
- Modify: `frontend/src/i18n/locales/en.ts`
- Modify: `frontend/src/i18n/locales/zh-TW.ts`

- [ ] **Step 1: Create the serialization module**

```typescript
// frontend/src/components/SubgraphEditor/graphSerialization.ts
import type { Node, Edge } from '@xyflow/react';
import { generateId } from '../../utils';

export interface PortDef {
  id: string;
  name: string;
}

export interface LayerNodeData {
  layerType: string;
  params: Record<string, any>;
  color: string;
  ports?: PortDef[]; // only Input / Output
  isMerge?: boolean;
  isBoundary?: boolean;
  [key: string]: unknown;
}

export interface GraphSpec {
  version: 2;
  nodes: Array<{
    id: string;
    type: string;
    params?: Record<string, any>;
    ports?: PortDef[];
    position?: { x: number; y: number };
  }>;
  edges: Array<{
    id: string;
    source: string;
    sourceHandle?: string | null;
    target: string;
    targetHandle?: string | null;
  }>;
}

const MERGE_TYPES = new Set(['Add', 'Concat', 'Multiply', 'Subtract', 'Mean', 'Stack']);

export function isMergeType(t: string): boolean {
  return MERGE_TYPES.has(t);
}

export function flowToGraphJson(nodes: Node<LayerNodeData>[], edges: Edge[]): string {
  const spec: GraphSpec = {
    version: 2,
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.layerType,
      params: n.data.params ?? {},
      ports: n.data.ports,
      position: n.position,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      sourceHandle: e.sourceHandle ?? null,
      target: e.target,
      targetHandle: e.targetHandle ?? null,
    })),
  };
  return JSON.stringify(spec);
}

export function graphToFlow(json: string): { nodes: Node<LayerNodeData>[]; edges: Edge[] } {
  let spec: GraphSpec;
  try {
    spec = JSON.parse(json);
  } catch {
    return emptyGraph();
  }
  if (spec.version !== 2 || !Array.isArray(spec.nodes) || !Array.isArray(spec.edges)) {
    return emptyGraph();
  }

  const nodes: Node<LayerNodeData>[] = spec.nodes.map((n) => {
    const isInput = n.type === 'Input';
    const isOutput = n.type === 'Output';
    const isBoundary = isInput || isOutput;
    const isMerge = isMergeType(n.type);

    return {
      id: n.id,
      type: isInput ? 'inputNode' : isOutput ? 'outputNode' : 'layerNode',
      position: n.position ?? { x: 0, y: 0 },
      data: {
        layerType: n.type,
        params: n.params ?? {},
        color: colorForType(n.type),
        ports: isBoundary ? n.ports : undefined,
        isMerge,
        isBoundary,
      },
    };
  });

  const edges: Edge[] = spec.edges.map((e) => ({
    id: e.id,
    source: e.source,
    sourceHandle: e.sourceHandle ?? undefined,
    target: e.target,
    targetHandle: e.targetHandle ?? undefined,
    style: { stroke: '#555', strokeWidth: 2 },
  }));

  return { nodes, edges };
}

export function emptyGraph(): { nodes: Node<LayerNodeData>[]; edges: Edge[] } {
  const inId = generateId();
  const outId = generateId();
  const inPortId = generateId();
  const outPortId = generateId();
  return {
    nodes: [
      {
        id: inId,
        type: 'inputNode',
        position: { x: 50, y: 50 },
        data: {
          layerType: 'Input',
          params: {},
          color: '#4CAF50',
          ports: [{ id: inPortId, name: 'x' }],
          isBoundary: true,
        },
      },
      {
        id: outId,
        type: 'outputNode',
        position: { x: 50, y: 400 },
        data: {
          layerType: 'Output',
          params: {},
          color: '#F44336',
          ports: [{ id: outPortId, name: 'out' }],
          isBoundary: true,
        },
      },
    ],
    edges: [],
  };
}

function colorForType(type: string): string {
  if (type === 'Input') return '#4CAF50';
  if (type === 'Output') return '#F44336';
  if (MERGE_TYPES.has(type)) return '#FF9800';
  // fallback colors by category — duplicated from SubgraphEditorModal LAYER_DEFS
  const colors: Record<string, string> = {
    Conv2d: '#4CAF50', Conv1d: '#4CAF50', ConvTranspose2d: '#4CAF50',
    BatchNorm2d: '#9C27B0', BatchNorm1d: '#9C27B0', LayerNorm: '#9C27B0',
    GroupNorm: '#9C27B0', InstanceNorm2d: '#9C27B0',
    MaxPool2d: '#2196F3', AvgPool2d: '#2196F3', AdaptiveAvgPool2d: '#2196F3',
    Dropout: '#FF9800',
    Linear: '#00BCD4', Embedding: '#00BCD4',
    Flatten: '#607D8B',
  };
  return colors[type] ?? '#F44336';
}

export interface ValidationError {
  message: string;
}

export function validateGraph(nodes: Node<LayerNodeData>[], edges: Edge[]): ValidationError | null {
  const inputs = nodes.filter((n) => n.data.layerType === 'Input');
  const outputs = nodes.filter((n) => n.data.layerType === 'Output');
  if (inputs.length !== 1) return { message: 'Graph must have exactly one Input node' };
  if (outputs.length !== 1) return { message: 'Graph must have exactly one Output node' };

  const input = inputs[0];
  const output = outputs[0];
  const inPorts = input.data.ports ?? [];
  const outPorts = output.data.ports ?? [];
  if (inPorts.length === 0) return { message: 'Input node must have at least one port' };
  if (outPorts.length === 0) return { message: 'Output node must have at least one port' };

  const inNames = inPorts.map((p) => p.name);
  if (new Set(inNames).size !== inNames.length) return { message: 'Input port names must be unique' };
  const outNames = outPorts.map((p) => p.name);
  if (new Set(outNames).size !== outNames.length) return { message: 'Output port names must be unique' };

  // Output port: exactly 1 incoming
  for (const p of outPorts) {
    const count = edges.filter((e) => e.target === output.id && e.targetHandle === p.id).length;
    if (count !== 1) {
      return { message: `Output port '${p.name}' must have exactly 1 incoming edge (got ${count})` };
    }
  }

  // Input port: at least 1 outgoing
  for (const p of inPorts) {
    const count = edges.filter((e) => e.source === input.id && e.sourceHandle === p.id).length;
    if (count < 1) {
      return { message: `Input port '${p.name}' is unused` };
    }
  }

  // Plain layers (non-merge, non-boundary): exactly 1 incoming
  for (const n of nodes) {
    if (n.data.isBoundary || n.data.isMerge) continue;
    const incoming = edges.filter((e) => e.target === n.id);
    if (incoming.length !== 1) {
      return { message: `Layer '${n.data.layerType}' must have exactly 1 incoming edge (got ${incoming.length})` };
    }
  }

  // Cycle check via Kahn's
  const inDegree: Record<string, number> = {};
  const outAdj: Record<string, string[]> = {};
  for (const n of nodes) {
    inDegree[n.id] = 0;
    outAdj[n.id] = [];
  }
  for (const e of edges) {
    inDegree[e.target] = (inDegree[e.target] ?? 0) + 1;
    outAdj[e.source] = [...(outAdj[e.source] ?? []), e.target];
  }
  const queue: string[] = Object.keys(inDegree).filter((k) => inDegree[k] === 0);
  let visited = 0;
  while (queue.length) {
    const id = queue.shift()!;
    visited++;
    for (const t of outAdj[id]) {
      inDegree[t]--;
      if (inDegree[t] === 0) queue.push(t);
    }
  }
  if (visited !== nodes.length) return { message: 'Graph contains a cycle' };

  return null;
}
```

- [ ] **Step 2: Add i18n strings**

In `frontend/src/i18n/locales/en.ts`, find the subgraph section and add:

```typescript
  'subgraph.category.io': 'I/O',
  'subgraph.category.merge': 'Merge',
  'subgraph.validation.cycle': 'Graph contains a cycle',
  'subgraph.validation.noInput': 'Graph must have exactly one Input node',
  'subgraph.validation.noOutput': 'Graph must have exactly one Output node',
  'subgraph.port.add': '+ Add port',
  'subgraph.port.remove': 'Remove',
  'subgraph.port.namePlaceholder': 'port name',
  'subgraph.port.duplicate': 'Duplicate port name',
  'subgraph.port.list': 'Ports',
```

Same keys (translated) in `frontend/src/i18n/locales/zh-TW.ts`:

```typescript
  'subgraph.category.io': '輸入/輸出',
  'subgraph.category.merge': '合併',
  'subgraph.validation.cycle': '圖形包含循環',
  'subgraph.validation.noInput': '圖形必須有一個 Input 節點',
  'subgraph.validation.noOutput': '圖形必須有一個 Output 節點',
  'subgraph.port.add': '+ 新增 port',
  'subgraph.port.remove': '移除',
  'subgraph.port.namePlaceholder': 'port 名稱',
  'subgraph.port.duplicate': 'port 名稱重複',
  'subgraph.port.list': 'Ports',
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd frontend && pnpm tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/SubgraphEditor/graphSerialization.ts frontend/src/i18n/locales/en.ts frontend/src/i18n/locales/zh-TW.ts
git commit -m "feat(frontend): add graph serialization + validation helpers"
```

---

### Task 8: InputNode component (dynamic output handles)

**Files:**
- Create: `frontend/src/components/SubgraphEditor/InputNode.tsx`

- [ ] **Step 1: Create the component**

```tsx
// frontend/src/components/SubgraphEditor/InputNode.tsx
import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { LayerNodeData } from './graphSerialization';

function InputNodeComponent({ data, selected }: NodeProps<LayerNodeData>) {
  const ports = data.ports ?? [];
  return (
    <div
      style={{
        background: '#1e1e1e',
        border: `2px solid ${selected ? '#fff' : '#4CAF5088'}`,
        borderRadius: 8,
        minWidth: 140,
        fontSize: '0.8125rem',
        color: '#eee',
        boxShadow: selected ? '0 0 12px #4CAF5044' : '0 3px 10px rgba(0,0,0,0.4)',
      }}
    >
      <div
        style={{
          background: '#4CAF50',
          padding: '5px 10px',
          borderRadius: '6px 6px 0 0',
          fontWeight: 600,
          fontSize: '0.8125rem',
          textAlign: 'center',
          color: '#fff',
        }}
      >
        Input
      </div>
      <div style={{ padding: '6px 10px 14px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {ports.map((p) => (
          <div key={p.id} style={{ fontSize: '0.6875rem', color: '#bbb', textAlign: 'center' }}>
            {p.name}
          </div>
        ))}
      </div>
      {ports.map((p, i) => {
        const left = ((i + 1) / (ports.length + 1)) * 100;
        return (
          <Handle
            key={p.id}
            id={p.id}
            type="source"
            position={Position.Bottom}
            style={{
              background: '#4CAF50',
              width: 10,
              height: 10,
              border: '2px solid #1e1e1e',
              left: `${left}%`,
              bottom: -5,
            }}
          />
        );
      })}
    </div>
  );
}

export const InputNode = memo(InputNodeComponent);
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && pnpm tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/SubgraphEditor/InputNode.tsx
git commit -m "feat(frontend): InputNode with dynamic output handles"
```

---

### Task 9: OutputNode component (dynamic input handles)

**Files:**
- Create: `frontend/src/components/SubgraphEditor/OutputNode.tsx`

- [ ] **Step 1: Create the component**

```tsx
// frontend/src/components/SubgraphEditor/OutputNode.tsx
import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { LayerNodeData } from './graphSerialization';

function OutputNodeComponent({ data, selected }: NodeProps<LayerNodeData>) {
  const ports = data.ports ?? [];
  return (
    <div
      style={{
        background: '#1e1e1e',
        border: `2px solid ${selected ? '#fff' : '#F4433688'}`,
        borderRadius: 8,
        minWidth: 140,
        fontSize: '0.8125rem',
        color: '#eee',
        boxShadow: selected ? '0 0 12px #F4433644' : '0 3px 10px rgba(0,0,0,0.4)',
      }}
    >
      {ports.map((p, i) => {
        const left = ((i + 1) / (ports.length + 1)) * 100;
        return (
          <Handle
            key={p.id}
            id={p.id}
            type="target"
            position={Position.Top}
            style={{
              background: '#F44336',
              width: 10,
              height: 10,
              border: '2px solid #1e1e1e',
              left: `${left}%`,
              top: -5,
            }}
          />
        );
      })}
      <div style={{ padding: '14px 10px 6px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {ports.map((p) => (
          <div key={p.id} style={{ fontSize: '0.6875rem', color: '#bbb', textAlign: 'center' }}>
            {p.name}
          </div>
        ))}
      </div>
      <div
        style={{
          background: '#F44336',
          padding: '5px 10px',
          borderRadius: '0 0 6px 6px',
          fontWeight: 600,
          fontSize: '0.8125rem',
          textAlign: 'center',
          color: '#fff',
        }}
      >
        Output
      </div>
    </div>
  );
}

export const OutputNode = memo(OutputNodeComponent);
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && pnpm tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/SubgraphEditor/OutputNode.tsx
git commit -m "feat(frontend): OutputNode with dynamic input handles"
```

---

### Task 10: PortListEditor component

**Files:**
- Create: `frontend/src/components/SubgraphEditor/PortListEditor.tsx`

- [ ] **Step 1: Create the component**

```tsx
// frontend/src/components/SubgraphEditor/PortListEditor.tsx
import { useI18n } from '../../i18n';
import { generateId } from '../../utils';
import type { Node, Edge } from '@xyflow/react';
import type { LayerNodeData, PortDef } from './graphSerialization';

interface Props {
  node: Node<LayerNodeData>;
  edges: Edge[];
  onUpdatePorts: (nodeId: string, ports: PortDef[]) => void;
  onRemoveEdges: (edgeIds: string[]) => void;
}

export function PortListEditor({ node, edges, onUpdatePorts, onRemoveEdges }: Props) {
  const { t } = useI18n();
  const ports = node.data.ports ?? [];
  const isInput = node.data.layerType === 'Input';

  const setName = (portId: string, name: string) => {
    const next = ports.map((p) => (p.id === portId ? { ...p, name } : p));
    onUpdatePorts(node.id, next);
  };

  const addPort = () => {
    const next: PortDef[] = [...ports, { id: generateId(), name: `port${ports.length + 1}` }];
    onUpdatePorts(node.id, next);
  };

  const removePort = (portId: string) => {
    const next = ports.filter((p) => p.id !== portId);
    onUpdatePorts(node.id, next);
    // Remove any edges referencing this port
    const orphaned = edges
      .filter((e) =>
        isInput
          ? e.source === node.id && e.sourceHandle === portId
          : e.target === node.id && e.targetHandle === portId
      )
      .map((e) => e.id);
    if (orphaned.length > 0) onRemoveEdges(orphaned);
  };

  const names = ports.map((p) => p.name);
  const hasDuplicate = (name: string) => names.filter((n) => n === name).length > 1;

  return (
    <div style={{ padding: '12px 10px' }}>
      <div
        style={{
          fontSize: '0.9375rem',
          fontWeight: 700,
          color: '#eee',
          marginBottom: 12,
          paddingBottom: 8,
          borderBottom: '1px solid #333',
        }}
      >
        {node.data.layerType} — {t('subgraph.port.list')}
      </div>
      {ports.map((p) => (
        <div key={p.id} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
          <input
            type="text"
            value={p.name}
            placeholder={t('subgraph.port.namePlaceholder')}
            onChange={(e) => setName(p.id, e.target.value)}
            title={hasDuplicate(p.name) ? t('subgraph.port.duplicate') : undefined}
            style={{
              flex: 1,
              padding: '5px 8px',
              background: '#222',
              border: hasDuplicate(p.name) ? '1px solid #F44336' : '1px solid #444',
              borderRadius: 4,
              color: '#ddd',
              fontSize: '0.8125rem',
              outline: 'none',
            }}
          />
          <button
            onClick={() => removePort(p.id)}
            disabled={ports.length === 1}
            style={{
              padding: '3px 8px',
              background: '#3a1515',
              border: '1px solid #F44336',
              borderRadius: 4,
              color: '#F44336',
              fontSize: '0.6875rem',
              cursor: ports.length === 1 ? 'not-allowed' : 'pointer',
              opacity: ports.length === 1 ? 0.4 : 1,
            }}
          >
            {t('subgraph.port.remove')}
          </button>
        </div>
      ))}
      <button
        onClick={addPort}
        style={{
          marginTop: 8,
          padding: '5px 10px',
          background: '#2a2a2a',
          border: '1px solid #444',
          borderRadius: 4,
          color: '#aaa',
          fontSize: '0.75rem',
          cursor: 'pointer',
          width: '100%',
        }}
      >
        {t('subgraph.port.add')}
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && pnpm tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/SubgraphEditor/PortListEditor.tsx
git commit -m "feat(frontend): PortListEditor for Input/Output nodes"
```

---

### Task 11: Rewrite SubgraphEditorModal — free-form edges, Merge palette, validation, custom node types

**Files:**
- Modify: `frontend/src/components/SubgraphEditor/SubgraphEditorModal.tsx`

- [ ] **Step 1: Add Merge layer defs and update LAYER_DEFS**

In `SubgraphEditorModal.tsx`, after the existing `LAYER_DEFS` array (line ~122), add merge layer definitions:

```typescript
const MERGE_LAYER_DEFS: LayerDef[] = [
  { type: 'Add', category: 'Merge', color: '#FF9800', params: [] },
  { type: 'Concat', category: 'Merge', color: '#FF9800', params: [p_int('dim', 1, 'Concatenation dim')] },
  { type: 'Multiply', category: 'Merge', color: '#FF9800', params: [] },
  { type: 'Subtract', category: 'Merge', color: '#FF9800', params: [] },
  { type: 'Mean', category: 'Merge', color: '#FF9800', params: [] },
  { type: 'Stack', category: 'Merge', color: '#FF9800', params: [p_int('dim', 1, 'Stack dim')] },
];

const ALL_LAYER_DEFS: LayerDef[] = [...LAYER_DEFS, ...MERGE_LAYER_DEFS];
const LAYER_DEF_MAP_FULL = new Map(ALL_LAYER_DEFS.map((d) => [d.type, d]));
```

Replace existing `LAYER_DEF_MAP` references with `LAYER_DEF_MAP_FULL` throughout the file.

- [ ] **Step 2: Replace nodeTypes and serialization imports**

Replace the existing `nodeTypes` constant (line ~378):

```typescript
import { InputNode } from './InputNode';
import { OutputNode } from './OutputNode';
import { PortListEditor } from './PortListEditor';
import {
  graphToFlow,
  flowToGraphJson,
  emptyGraph,
  validateGraph,
  isMergeType,
  type LayerNodeData as GraphLayerNodeData,
  type PortDef,
} from './graphSerialization';

const nodeTypes: NodeTypes = {
  layerNode: LayerNode,
  inputNode: InputNode,
  outputNode: OutputNode,
};
```

Remove the local `LayerNodeData` interface (around line 128) — replace usages with `GraphLayerNodeData`.

- [ ] **Step 3: Replace `layersToFlow` / `flowToLayersJson` with `graphToFlow` / `flowToGraphJson`**

Delete the existing `layersToFlow` and `flowToLayersJson` functions entirely (lines ~135-224). They are replaced by the helpers in `graphSerialization.ts`.

Inside `SubgraphFlowInner`, change the `initial` derivation:

```typescript
  const initial = useMemo(() => {
    const parsed = graphToFlow(initialLayersJson);
    if (parsed.nodes.length === 0) return emptyGraph();
    return parsed;
  }, [initialLayersJson]);
```

- [ ] **Step 4: Remove `rebuildChain` and Y-position auto-chaining**

Delete the `rebuildChain` callback entirely (lines ~422-434). In `handleNodesChange`, remove the `hasDrag` / `rebuildChain` block — it becomes:

```typescript
  const handleNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setNodes((prev) => applyNodeChanges(changes, prev) as Node<GraphLayerNodeData>[]);
    },
    []
  );
```

In `addLayer`, remove the `setTimeout(() => rebuildChain(updated), 0)` call. Just append the node:

```typescript
  const addLayer = useCallback(
    (layerType: string, position: { x: number; y: number }) => {
      const def = LAYER_DEF_MAP_FULL.get(layerType);
      if (!def) return;
      const defaultParams: Record<string, any> = {};
      for (const p of def.params) defaultParams[p.name] = p.default;
      const newNode: Node<GraphLayerNodeData> = {
        id: generateId(),
        type: 'layerNode',
        position,
        data: {
          layerType,
          params: defaultParams,
          color: def.color,
          isMerge: isMergeType(layerType),
          isBoundary: false,
        },
      };
      setNodes((prev) => [...prev, newNode]);
    },
    []
  );
```

In `handleDeleteLayer`, remove the rebuildChain call:

```typescript
  const handleDeleteLayer = useCallback((nodeId: string) => {
    setNodes((prev) => prev.filter((n) => n.id !== nodeId));
    setEdges((prev) => prev.filter((e) => e.source !== nodeId && e.target !== nodeId));
    setSelectedNodeId((prev) => (prev === nodeId ? null : prev));
  }, []);
```

- [ ] **Step 5: Add `onConnect` handler with validity check**

Add to `SubgraphFlowInner` after `onEdgesChange`:

```typescript
  const isValidConnection = useCallback(
    (conn: any) => {
      // Plain layer input handle: reject if already has incoming
      const targetNode = nodes.find((n) => n.id === conn.target);
      if (!targetNode) return false;
      if (!targetNode.data.isMerge && !targetNode.data.isBoundary) {
        const existing = edges.filter((e) => e.target === conn.target);
        if (existing.length >= 1) return false;
      }
      // Output port: reject if already has incoming for that handle
      if (targetNode.data.layerType === 'Output' && conn.targetHandle) {
        const existing = edges.filter((e) => e.target === conn.target && e.targetHandle === conn.targetHandle);
        if (existing.length >= 1) return false;
      }
      return true;
    },
    [nodes, edges]
  );

  const onConnect = useCallback(
    (params: any) => {
      setEdges((eds) => [
        ...eds,
        {
          id: generateId(),
          source: params.source,
          sourceHandle: params.sourceHandle ?? undefined,
          target: params.target,
          targetHandle: params.targetHandle ?? undefined,
          style: { stroke: '#555', strokeWidth: 2 },
        },
      ]);
    },
    []
  );
```

Pass `onConnect={onConnect}` and `isValidConnection={isValidConnection}` to the `<ReactFlow>` element.

- [ ] **Step 6: Add port management handlers + render PortListEditor when boundary node selected**

Add to `SubgraphFlowInner`:

```typescript
  const handleUpdatePorts = useCallback((nodeId: string, ports: PortDef[]) => {
    setNodes((prev) =>
      prev.map((n) => (n.id === nodeId ? { ...n, data: { ...n.data, ports } } : n))
    );
  }, []);

  const handleRemoveEdges = useCallback((edgeIds: string[]) => {
    const idSet = new Set(edgeIds);
    setEdges((prev) => prev.filter((e) => !idSet.has(e.id)));
  }, []);
```

Replace the right-side param panel `selectedNode ? <ParamEditor> : <empty>` block with:

```tsx
              {selectedNode ? (
                selectedNode.data.isBoundary ? (
                  <PortListEditor
                    node={selectedNode}
                    edges={edges}
                    onUpdatePorts={handleUpdatePorts}
                    onRemoveEdges={handleRemoveEdges}
                  />
                ) : (
                  <ParamEditor
                    node={selectedNode}
                    onParamChange={handleParamChange}
                    onDelete={handleDeleteLayer}
                  />
                )
              ) : (
                <div
                  style={{
                    padding: '20px 10px',
                    textAlign: 'center',
                    color: '#555',
                    fontSize: '0.75rem',
                  }}
                >
                  {t('subgraph.noParams')}
                </div>
              )}
```

- [ ] **Step 7: Add validation on Apply**

Replace `handleApply`:

```typescript
  const handleApply = () => {
    const err = validateGraph(nodes, edges);
    if (err) {
      useToastStore.getState().addToast(err.message, 'error');
      return;
    }
    onApply(flowToGraphJson(nodes, edges));
  };
```

- [ ] **Step 8: Wire export/import to v2 schema**

Replace `handleExport`:

```typescript
  const handleExport = () => {
    const json = flowToGraphJson(nodes, edges);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model_architecture.json';
    a.click();
    URL.revokeObjectURL(url);
  };
```

Replace `handleFileSelect`:

```typescript
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const { nodes: newNodes, edges: newEdges } = graphToFlow(text);
        if (newNodes.length === 0) throw new Error('Empty or invalid v2 graph');
        setNodes(newNodes);
        setEdges(newEdges);
        setSelectedNodeId(null);
        setTimeout(() => fitView({ padding: 0.3 }), 50);
      } catch (err) {
        useToastStore.getState().addToast(t('subgraph.import.fail', { error: String(err) }), 'error');
      }
    };
    reader.readAsText(file);
    event.target.value = '';
  };
```

- [ ] **Step 9: Verify TypeScript compiles + dev server starts cleanly**

Run: `cd frontend && pnpm tsc --noEmit`
Expected: No errors

- [ ] **Step 10: Commit**

```bash
git add frontend/src/components/SubgraphEditor/SubgraphEditorModal.tsx
git commit -m "feat(frontend): rewrite SubgraphEditorModal as free-form DAG editor"
```

---

### Task 12: Chrome smoke test — fresh model, residual block, validation

- [ ] **Step 1: Open the app in Chrome and create a new tab**

Use `mcp__claude-in-chrome__navigate` to load `http://localhost:5173`. Open a new in-app tab so the canvas is empty.

- [ ] **Step 2: Drag a SequentialModel onto the canvas, double-click to open the editor**

Drag from the Training category. Double-click the node. Verify the modal opens, showing pre-populated Input and Output nodes plus the default CNN classifier (because the default param value is the v2 spec).

- [ ] **Step 3: Verify the structure renders as a DAG**

Confirm:
- Input node has a single bottom handle labeled `x`
- Output node has a single top handle labeled `y`
- Layers are connected by edges (not auto-chained from Y position)

- [ ] **Step 4: Build a residual block from scratch**

Clear the existing layers (delete all middle nodes). Drag from palette to construct:
`Input(x) → Linear(4,4) → ReLU → Linear(4,4) → Add ← Input(x) (skip) → Output(y)`

The Add node should accept 2 incoming edges. Verify connections work.

- [ ] **Step 5: Click Apply, verify the validation passes and modal closes**

- [ ] **Step 6: Test cycle validation**

Reopen the editor. Create a cycle by connecting an output handle back to an upstream node. Click Apply.
Expected: Toast "Graph contains a cycle". Modal stays open.

- [ ] **Step 7: Test removing a port also removes its edges**

Select Input node, add a 2nd port `y`, connect it somewhere, then delete that port via PortListEditor. Verify the edge is gone.

- [ ] **Step 8: No commit (smoke test only). Document anything that broke and fix in a follow-up commit before proceeding.**

---

## Phase 3 — Examples Rewrite (one at a time, Chrome verification each)

For each task in this phase, the steps are the same template:

- [ ] **Step A: Rewrite the example's `graph.json`**, replacing the old flat-array `layers` string with a v2 spec JSON string. Use stable string IDs for nodes and ports. Maintain the example's existing high-level structure.
- [ ] **Step B: Reload the frontend** (Vite HMR usually picks up backend example changes via the API).
- [ ] **Step C: Open the example from the Empty Canvas overlay or Examples menu in Chrome.**
- [ ] **Step D: Double-click the SequentialModel node**, verify the graph renders correctly.
- [ ] **Step E: Click "Run Pipeline"** (or "執行管線"). Verify execution completes without error in the execution log. For Model_Architecture-only examples (no Dataset), confirm the model builds (param count appears in logs); for Usage_Example, confirm at least one batch / forward pass runs.
- [ ] **Step F: Commit.**

### Task 13: Rewrite TrainCNN-MNIST

**Files:**
- Modify: `examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json`

**Layers spec to substitute** (the inner `params.layers` string of the SequentialModel node `model-builder`):

```json
{
  "version": 2,
  "nodes": [
    {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
    {"id": "c1", "type": "Conv2d", "params": {"in_channels": 1, "out_channels": 32, "kernel_size": 3, "padding": 1}},
    {"id": "r1", "type": "ReLU"},
    {"id": "p1", "type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
    {"id": "c2", "type": "Conv2d", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1}},
    {"id": "r2", "type": "ReLU"},
    {"id": "p2", "type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
    {"id": "f", "type": "Flatten"},
    {"id": "l1", "type": "Linear", "params": {"in_features": 3136, "out_features": 128}},
    {"id": "r3", "type": "ReLU"},
    {"id": "l2", "type": "Linear", "params": {"in_features": 128, "out_features": 10}},
    {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]}
  ],
  "edges": [
    {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "c1"},
    {"id": "e2", "source": "c1", "target": "r1"},
    {"id": "e3", "source": "r1", "target": "p1"},
    {"id": "e4", "source": "p1", "target": "c2"},
    {"id": "e5", "source": "c2", "target": "r2"},
    {"id": "e6", "source": "r2", "target": "p2"},
    {"id": "e7", "source": "p2", "target": "f"},
    {"id": "e8", "source": "f", "target": "l1"},
    {"id": "e9", "source": "l1", "target": "r3"},
    {"id": "e10", "source": "r3", "target": "l2"},
    {"id": "e11", "source": "l2", "target": "out", "targetHandle": "p_y"}
  ]
}
```

This must be embedded as a JSON-string value in the existing `model-builder.data.params.layers` slot (escape quotes). Run steps A–F. Verify the training pipeline completes ≥1 batch and the ModelSaver writes weights.

```bash
git add examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
git commit -m "feat(examples): migrate TrainCNN-MNIST to v2 graph spec"
```

### Task 14: Rewrite InferenceCNN-MNIST

Same architecture as Task 13 (must match exactly so weights load). Same steps A–F. Verify inference produces logits + softmax output.

```bash
git add examples/Usage_Example/CNN-MNIST/InferenceCNN-MNIST/graph.json
git commit -m "feat(examples): migrate InferenceCNN-MNIST to v2 graph spec"
```

### Task 15: Rewrite ResNet-SkipConnection-CNN with REAL residual blocks

**Files:**
- Modify: `examples/Model_Architecture/ResNet-SkipConnection-CNN/graph.json`

The new spec must contain real `Add` skip connections. Build 2 residual blocks of:
```
Input(x) → stem(Conv7x7+BN+ReLU+Pool)
       → block1: in → Conv3x3 → BN → ReLU → Conv3x3 → BN → Add(skip=in) → ReLU
       → block2: same shape
       → AdaptiveAvgPool → Flatten → Linear → Output(y)
```

Concrete v2 spec to embed (channels = 64, input image 3x224x224):

```json
{
  "version": 2,
  "nodes": [
    {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
    {"id": "stem_conv", "type": "Conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3}},
    {"id": "stem_bn", "type": "BatchNorm2d", "params": {"num_features": 64}},
    {"id": "stem_relu", "type": "ReLU"},
    {"id": "stem_pool", "type": "MaxPool2d", "params": {"kernel_size": 3, "stride": 2}},

    {"id": "b1_c1", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
    {"id": "b1_bn1", "type": "BatchNorm2d", "params": {"num_features": 64}},
    {"id": "b1_r1", "type": "ReLU"},
    {"id": "b1_c2", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
    {"id": "b1_bn2", "type": "BatchNorm2d", "params": {"num_features": 64}},
    {"id": "b1_add", "type": "Add"},
    {"id": "b1_r2", "type": "ReLU"},

    {"id": "b2_c1", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
    {"id": "b2_bn1", "type": "BatchNorm2d", "params": {"num_features": 64}},
    {"id": "b2_r1", "type": "ReLU"},
    {"id": "b2_c2", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
    {"id": "b2_bn2", "type": "BatchNorm2d", "params": {"num_features": 64}},
    {"id": "b2_add", "type": "Add"},
    {"id": "b2_r2", "type": "ReLU"},

    {"id": "head_pool", "type": "AdaptiveAvgPool2d", "params": {"output_size": 1}},
    {"id": "head_flat", "type": "Flatten"},
    {"id": "head_fc", "type": "Linear", "params": {"in_features": 64, "out_features": 1000}},
    {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]}
  ],
  "edges": [
    {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "stem_conv"},
    {"id": "e2", "source": "stem_conv", "target": "stem_bn"},
    {"id": "e3", "source": "stem_bn", "target": "stem_relu"},
    {"id": "e4", "source": "stem_relu", "target": "stem_pool"},

    {"id": "e5", "source": "stem_pool", "target": "b1_c1"},
    {"id": "e6", "source": "b1_c1", "target": "b1_bn1"},
    {"id": "e7", "source": "b1_bn1", "target": "b1_r1"},
    {"id": "e8", "source": "b1_r1", "target": "b1_c2"},
    {"id": "e9", "source": "b1_c2", "target": "b1_bn2"},
    {"id": "e10", "source": "b1_bn2", "target": "b1_add"},
    {"id": "e11", "source": "stem_pool", "target": "b1_add"},
    {"id": "e12", "source": "b1_add", "target": "b1_r2"},

    {"id": "e13", "source": "b1_r2", "target": "b2_c1"},
    {"id": "e14", "source": "b2_c1", "target": "b2_bn1"},
    {"id": "e15", "source": "b2_bn1", "target": "b2_r1"},
    {"id": "e16", "source": "b2_r1", "target": "b2_c2"},
    {"id": "e17", "source": "b2_c2", "target": "b2_bn2"},
    {"id": "e18", "source": "b2_bn2", "target": "b2_add"},
    {"id": "e19", "source": "b1_r2", "target": "b2_add"},
    {"id": "e20", "source": "b2_add", "target": "b2_r2"},

    {"id": "e21", "source": "b2_r2", "target": "head_pool"},
    {"id": "e22", "source": "head_pool", "target": "head_flat"},
    {"id": "e23", "source": "head_flat", "target": "head_fc"},
    {"id": "e24", "source": "head_fc", "target": "out", "targetHandle": "p_y"}
  ]
}
```

Steps A–F. Verify the model builds (Print node shows the GraphModelModule with the right number of params: ~3.5M for this configuration).

```bash
git add examples/Model_Architecture/ResNet-SkipConnection-CNN/graph.json
git commit -m "feat(examples): rewrite ResNet with real residual blocks (Add skip)"
```

### Task 16: NEW UNet-Segmentation-CNN example

**Files:**
- Create: `examples/Model_Architecture/UNet-Segmentation-CNN/graph.json`

Build a small 2-level UNet (encoder → bottleneck → decoder with concat skip):
- Input → Conv(3→16) → ReLU → Conv(16→16) → ReLU → save as `enc1`
- Pool → Conv(16→32) → ReLU → Conv(32→32) → ReLU → save as `enc2`
- Pool → Conv(32→64) → ReLU → Conv(64→64) → ReLU (bottleneck)
- Upsample → ConvTranspose2d(64→32) → Concat with enc2 → Conv(64→32) → ReLU → Conv(32→32) → ReLU
- Upsample → ConvTranspose2d(32→16) → Concat with enc1 → Conv(32→16) → ReLU → Conv(16→16) → ReLU
- Conv(16→1) → Output

The full graph.json:

```json
{
  "name": "UNet for Image Segmentation",
  "description": "Classic 2-level UNet showing encoder-decoder architecture with Concat-based skip connections from encoder features into the decoder.",
  "nodes": [
    {
      "id": "model-builder",
      "type": "SequentialModel",
      "position": { "x": 50, "y": 200 },
      "data": {
        "params": {
          "layers": "<<<INSERT V2 JSON STRING HERE>>>"
        }
      }
    },
    {
      "id": "print-model",
      "type": "Print",
      "position": { "x": 500, "y": 200 },
      "data": { "params": { "label": "UNet model" } }
    }
  ],
  "edges": [
    { "id": "e1", "source": "model-builder", "target": "print-model", "sourceHandle": "model", "targetHandle": "value" }
  ]
}
```

The v2 spec to embed (escape and inline as the `layers` string):

```json
{
  "version": 2,
  "nodes": [
    {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},

    {"id": "e1c1", "type": "Conv2d", "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1}},
    {"id": "e1r1", "type": "ReLU"},
    {"id": "e1c2", "type": "Conv2d", "params": {"in_channels": 16, "out_channels": 16, "kernel_size": 3, "padding": 1}},
    {"id": "e1r2", "type": "ReLU"},

    {"id": "p1", "type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},

    {"id": "e2c1", "type": "Conv2d", "params": {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "padding": 1}},
    {"id": "e2r1", "type": "ReLU"},
    {"id": "e2c2", "type": "Conv2d", "params": {"in_channels": 32, "out_channels": 32, "kernel_size": 3, "padding": 1}},
    {"id": "e2r2", "type": "ReLU"},

    {"id": "p2", "type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},

    {"id": "bnc1", "type": "Conv2d", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1}},
    {"id": "bnr1", "type": "ReLU"},
    {"id": "bnc2", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
    {"id": "bnr2", "type": "ReLU"},

    {"id": "u2", "type": "ConvTranspose2d", "params": {"in_channels": 64, "out_channels": 32, "kernel_size": 2, "stride": 2}},
    {"id": "cat2", "type": "Concat", "params": {"dim": 1}},
    {"id": "d2c1", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 32, "kernel_size": 3, "padding": 1}},
    {"id": "d2r1", "type": "ReLU"},
    {"id": "d2c2", "type": "Conv2d", "params": {"in_channels": 32, "out_channels": 32, "kernel_size": 3, "padding": 1}},
    {"id": "d2r2", "type": "ReLU"},

    {"id": "u1", "type": "ConvTranspose2d", "params": {"in_channels": 32, "out_channels": 16, "kernel_size": 2, "stride": 2}},
    {"id": "cat1", "type": "Concat", "params": {"dim": 1}},
    {"id": "d1c1", "type": "Conv2d", "params": {"in_channels": 32, "out_channels": 16, "kernel_size": 3, "padding": 1}},
    {"id": "d1r1", "type": "ReLU"},
    {"id": "d1c2", "type": "Conv2d", "params": {"in_channels": 16, "out_channels": 16, "kernel_size": 3, "padding": 1}},
    {"id": "d1r2", "type": "ReLU"},

    {"id": "head", "type": "Conv2d", "params": {"in_channels": 16, "out_channels": 1, "kernel_size": 1}},

    {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]}
  ],
  "edges": [
    {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "e1c1"},
    {"id": "e2", "source": "e1c1", "target": "e1r1"},
    {"id": "e3", "source": "e1r1", "target": "e1c2"},
    {"id": "e4", "source": "e1c2", "target": "e1r2"},
    {"id": "e5", "source": "e1r2", "target": "p1"},

    {"id": "e6", "source": "p1", "target": "e2c1"},
    {"id": "e7", "source": "e2c1", "target": "e2r1"},
    {"id": "e8", "source": "e2r1", "target": "e2c2"},
    {"id": "e9", "source": "e2c2", "target": "e2r2"},
    {"id": "e10", "source": "e2r2", "target": "p2"},

    {"id": "e11", "source": "p2", "target": "bnc1"},
    {"id": "e12", "source": "bnc1", "target": "bnr1"},
    {"id": "e13", "source": "bnr1", "target": "bnc2"},
    {"id": "e14", "source": "bnc2", "target": "bnr2"},

    {"id": "e15", "source": "bnr2", "target": "u2"},
    {"id": "e16", "source": "u2", "target": "cat2"},
    {"id": "e17", "source": "e2r2", "target": "cat2"},
    {"id": "e18", "source": "cat2", "target": "d2c1"},
    {"id": "e19", "source": "d2c1", "target": "d2r1"},
    {"id": "e20", "source": "d2r1", "target": "d2c2"},
    {"id": "e21", "source": "d2c2", "target": "d2r2"},

    {"id": "e22", "source": "d2r2", "target": "u1"},
    {"id": "e23", "source": "u1", "target": "cat1"},
    {"id": "e24", "source": "e1r2", "target": "cat1"},
    {"id": "e25", "source": "cat1", "target": "d1c1"},
    {"id": "e26", "source": "d1c1", "target": "d1r1"},
    {"id": "e27", "source": "d1r1", "target": "d1c2"},
    {"id": "e28", "source": "d1c2", "target": "d1r2"},

    {"id": "e29", "source": "d1r2", "target": "head"},
    {"id": "e30", "source": "head", "target": "out", "targetHandle": "p_y"}
  ]
}
```

Steps A–F. Run pipeline; confirm model builds and Print shows GraphModelModule.

```bash
git add examples/Model_Architecture/UNet-Segmentation-CNN/graph.json
git commit -m "feat(examples): add UNet showcase with Concat skip connections"
```

### Task 17–28: Rewrite remaining 12 Model_Architecture examples

For each example, follow steps A–F. The conversion is mechanical (linear DAG, Option Y — wrapper layers stay opaque). For each: open the existing `graph.json`, identify the inner SequentialModel `params.layers` flat array, and convert to a v2 spec with:
- One `Input` node `{id:"in", type:"Input", ports:[{id:"p_x", name:"x"}]}`
- Each existing layer becomes a node with `id: "n{i}"`, `type: <existing type>`, `params: <existing params minus 'type'>`
- One `Output` node `{id:"out", type:"Output", ports:[{id:"p_y", name:"y"}]}`
- Edges: `in → n0 → n1 → ... → n{N-1} → out`, each with stable id `e{i}`

After each, run Chrome verification (steps C–E above), then commit.

- [ ] **Task 17:** `examples/Model_Architecture/ConvNeXt-CNN/graph.json` — commit `feat(examples): migrate ConvNeXt-CNN to v2 graph spec`
- [ ] **Task 18:** `examples/Model_Architecture/EfficientNet-CNN/graph.json` — commit `feat(examples): migrate EfficientNet-CNN to v2 graph spec`
- [ ] **Task 19:** `examples/Model_Architecture/ViT-ImageClassifier-Transformer/graph.json` — commit `feat(examples): migrate ViT to v2 graph spec`
- [ ] **Task 20:** `examples/Model_Architecture/BERT-Encoder-Transformer/graph.json` — commit `feat(examples): migrate BERT to v2 graph spec`
- [ ] **Task 21:** `examples/Model_Architecture/GPT-DecoderOnly-Transformer/graph.json` — commit `feat(examples): migrate GPT to v2 graph spec`
- [ ] **Task 22:** `examples/Model_Architecture/LLaMA-Decoder-Transformer/graph.json` — commit `feat(examples): migrate LLaMA to v2 graph spec`
- [ ] **Task 23:** `examples/Model_Architecture/SwinTransformer-Transformer/graph.json` — commit `feat(examples): migrate Swin to v2 graph spec`
- [ ] **Task 24:** `examples/Model_Architecture/DiT-Diffusion-Transformer/graph.json` — commit `feat(examples): migrate DiT to v2 graph spec`
- [ ] **Task 25:** `examples/Model_Architecture/Seq2Seq-Attention-RNN/graph.json` — commit `feat(examples): migrate Seq2Seq to v2 graph spec`
- [ ] **Task 26:** `examples/Model_Architecture/BiGRU-SpeechRecognition-RNN/graph.json` — commit `feat(examples): migrate BiGRU to v2 graph spec`
- [ ] **Task 27:** `examples/Model_Architecture/TimeSeries-LSTM-RNN/graph.json` — commit `feat(examples): migrate TimeSeries-LSTM to v2 graph spec`
- [ ] **Task 28:** `examples/Model_Architecture/DQN-Atari-RL/graph.json` AND `examples/Model_Architecture/PPO-Robotics-RL/graph.json` (combined commit since both are small RL examples) — commit `feat(examples): migrate DQN and PPO to v2 graph spec`

---

## Phase 4 — Final Verification

### Task 29: Full backend test pass

- [ ] **Step 1: Run all backend tests**

Run: `cd backend && pytest tests/ -v`
Expected: PASS (existing tests + new test_graph_model.py)

- [ ] **Step 2: If anything fails, fix and re-run**

### Task 30: Sanity-check all 17 examples load via the API

- [ ] **Step 1: Hit the examples/list endpoint**

Run: `curl -s http://127.0.0.1:8000/api/examples/list | python -c "import sys,json; ex = json.load(sys.stdin); print(len(ex), 'examples'); [print(e['path']) for e in ex]"`
Expected: 17 examples listed (16 migrated + 1 new UNet)

- [ ] **Step 2: For each example, fetch via /api/examples/load and parse the layers JSON**

Run: `for path in <paths>; do curl -s "http://127.0.0.1:8000/api/examples/load?path=$path" | python -c "import sys,json; d=json.load(sys.stdin); sm=[n for n in d['nodes'] if n['type']=='SequentialModel']; assert sm, f'no SequentialModel in $path'; spec=json.loads(sm[0]['data']['params']['layers']); assert spec.get('version')==2, f'$path not v2'; print('$path OK')"; done`
Expected: every example prints "OK"

- [ ] **Step 3: Final commit if any fix-ups were needed**

```bash
git add -A
git commit -m "chore: final verification fixes for graph model migration"
```

---

## Self-Review Checklist

(Run by the plan author after writing — fix inline.)

- ✅ Spec §4 (DAG schema) → Tasks 1, 7
- ✅ Spec §4.1 (validation rules) → Tasks 5 (backend), 7 (frontend), 11 (Apply)
- ✅ Spec §5.1 (graph_model.py module) → Task 1
- ✅ Spec §5.2 (forward semantics) → Task 1, tested in Tasks 2, 3, 4
- ✅ Spec §5.3 (merge layers) → Task 1
- ✅ Spec §5.4 (backend validation) → Task 1, tested in Task 5
- ✅ Spec §6.1 (file structure) → Tasks 7–11
- ✅ Spec §6.2 (editor behavior) → Task 11
- ✅ Spec §6.3 (serialization) → Task 7
- ✅ Spec §7 (rewrite plan + showcase) → Tasks 13–28
- ✅ Spec §7.1 (per-example verification) → Steps A–F template
- ✅ Spec §8 (testing) → Tasks 1–6, 12, 29
- ✅ Spec §9 (error handling) → Toast on validation errors in Task 11
- ✅ Spec §10 (rollout order) → Phase 1 → 2 → 3 → 4 ordering

No placeholders found. Type names consistent (`GraphModelModule`, `build_graph_model`, `LayerNodeData`, `PortDef`, `GraphSpec`, `validateGraph`).
