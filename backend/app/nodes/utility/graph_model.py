"""
GraphModelModule — DAG-based nn.Module builder for the v2 SequentialModel spec.

The v2 spec is a JSON object::

    {
        "version": 2,
        "nodes": [
            {"id": "in",  "type": "Input",  "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "n0",  "type": "Conv2d", "params": {"in_channels": 3, ...}},
            ...
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]}
        ],
        "edges": [
            {"id": "e1", "source": "in",  "sourceHandle": "p_x", "target": "n0"},
            {"id": "e2", "source": "n0",  "target": "n1"},
            ...
        ]
    }

``build_graph_model(spec_dict)`` parses the spec, builds all nn.Modules, and returns a
``GraphModelModule`` that can be used anywhere an ``nn.Sequential`` is expected.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Merge layer wrappers ────────────────────────────────────────


class _AddMerge(nn.Module):
    """Element-wise sum of any number of input tensors."""

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = tensors[0]
        for t in tensors[1:]:
            out = out + t
        return out


class _ConcatMerge(nn.Module):
    """Concatenate tensors along a given dimension."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.cat(tensors, dim=self.dim)


class _MultiplyMerge(nn.Module):
    """Element-wise product of any number of input tensors."""

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = tensors[0]
        for t in tensors[1:]:
            out = out * t
        return out


# ── Misc Sequential-compatible wrappers ────────────────────────

class _ReshapeMod(nn.Module):
    def __init__(self, shape: str):
        super().__init__()
        self._shape = [int(d) for d in shape.split(",")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), *self._shape)


class _SelectIndexMod(nn.Module):
    def __init__(self, dim: int = 1, index: int = 0):
        super().__init__()
        self._dim = dim
        self._index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.select(self._dim, self._index)


class _LSTMMod(nn.Module):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.lstm = nn.LSTM(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class _GRUMod(nn.Module):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.gru = nn.GRU(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


class _TransformerEncoderMod(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int = 1, dim_feedforward: int = 2048):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _TransformerDecoderMod(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int = 1, dim_feedforward: int = 2048):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x, x)


class _MultiHeadAttentionMod(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out


# ── Layer builder ───────────────────────────────────────────────

_STANDARD_BUILDERS: dict[str, type] = {
    "Conv2d": nn.Conv2d,
    "Conv1d": nn.Conv1d,
    "ConvTranspose2d": nn.ConvTranspose2d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm1d": nn.BatchNorm1d,
    "LayerNorm": nn.LayerNorm,
    "GroupNorm": nn.GroupNorm,
    "InstanceNorm2d": nn.InstanceNorm2d,
    "MaxPool2d": nn.MaxPool2d,
    "AvgPool2d": nn.AvgPool2d,
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
    "Dropout": nn.Dropout,
    "Linear": nn.Linear,
    "Flatten": nn.Flatten,
    "Embedding": nn.Embedding,
}

_WRAPPER_BUILDERS: dict[str, type] = {
    "Reshape": _ReshapeMod,
    "SelectIndex": _SelectIndexMod,
    "LSTM": _LSTMMod,
    "GRU": _GRUMod,
    "TransformerEncoder": _TransformerEncoderMod,
    "TransformerDecoder": _TransformerDecoderMod,
    "MultiHeadAttention": _MultiHeadAttentionMod,
}

_ACTIVATIONS: dict[str, nn.Module] = {
    "ReLU": nn.ReLU(inplace=True),
    "GELU": nn.GELU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "LeakyReLU": nn.LeakyReLU(inplace=True),
    "ELU": nn.ELU(inplace=True),
    "SiLU": nn.SiLU(inplace=True),
    "Mish": nn.Mish(inplace=True),
    "SELU": nn.SELU(inplace=True),
    "PReLU": nn.PReLU(),
    "Hardswish": nn.Hardswish(inplace=True),
    "Softmax": nn.Softmax(dim=-1),
}

_MERGE_BUILDERS: dict[str, type] = {
    "Add": _AddMerge,
    "Concat": _ConcatMerge,
    "Multiply": _MultiplyMerge,
}


def _build_node_module(node_cfg: dict[str, Any]) -> nn.Module | None:
    """Build an nn.Module from a v2 node config dict. Returns None for Input/Output sentinel nodes."""
    t = node_cfg.get("type", "")
    params = node_cfg.get("params", {})

    if t in ("Input", "Output"):
        return None

    if t in _ACTIVATIONS:
        # Return a fresh instance (not the shared one) to avoid inplace issues with branched graphs
        act_map: dict[str, Any] = {
            "ReLU": nn.ReLU(inplace=False),
            "GELU": nn.GELU(),
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "LeakyReLU": nn.LeakyReLU(inplace=False),
            "ELU": nn.ELU(inplace=False),
            "SiLU": nn.SiLU(inplace=False),
            "Mish": nn.Mish(inplace=False),
            "SELU": nn.SELU(inplace=False),
            "PReLU": nn.PReLU(),
            "Hardswish": nn.Hardswish(inplace=False),
            "Softmax": nn.Softmax(dim=-1),
        }
        if t in act_map:
            return act_map[t]

    if t in _MERGE_BUILDERS:
        return _MERGE_BUILDERS[t](**params)

    if t in _WRAPPER_BUILDERS:
        return _WRAPPER_BUILDERS[t](**params)

    if t in _STANDARD_BUILDERS:
        return _STANDARD_BUILDERS[t](**params)

    raise ValueError(f"GraphModel: unknown layer type '{t}'")


# ── GraphModelModule ─────────────────────────────────────────────

class GraphModelModule(nn.Module):
    """
    A DAG-based model that executes layers in topological order.

    Parameters
    ----------
    node_order : list[str]
        Topologically sorted list of node IDs (Input and Output included for bookkeeping).
    modules_dict : dict[str, nn.Module]
        Map from node ID → nn.Module for all non-sentinel nodes.
    in_edges : dict[str, list[str]]
        Map from node ID → list of upstream node IDs (in order).
    input_node_id : str
        ID of the Input sentinel node.
    output_node_id : str
        ID of the Output sentinel node.
    """

    def __init__(
        self,
        node_order: list[str],
        modules_dict: dict[str, nn.Module],
        in_edges: dict[str, list[str]],
        input_node_id: str,
        output_node_id: str,
    ):
        super().__init__()
        self._node_order = node_order
        self._in_edges = in_edges
        self._input_node_id = input_node_id
        self._output_node_id = output_node_id
        # Register all sub-modules under their IDs (replacing '-' / '.' with '_' for valid attr names)
        for node_id, mod in modules_dict.items():
            self.add_module(node_id.replace("-", "_").replace(".", "_"), mod)
        # Also store a plain dict for forward() lookup
        self._mods: dict[str, nn.Module] = dict(modules_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cache: dict[str, torch.Tensor] = {}
        cache[self._input_node_id] = x

        for node_id in self._node_order:
            if node_id == self._input_node_id:
                continue
            if node_id == self._output_node_id:
                # Collect all inputs to output node
                preds = self._in_edges.get(node_id, [])
                if not preds:
                    raise RuntimeError("Output node has no input edges")
                return cache[preds[0]]

            preds = self._in_edges.get(node_id, [])
            mod = self._mods[node_id]

            if len(preds) == 0:
                raise RuntimeError(f"Node '{node_id}' has no input edges but is not the Input node")
            elif len(preds) == 1:
                out = mod(cache[preds[0]])
            else:
                # Multi-input node (merge node like Add/Concat)
                inputs = [cache[p] for p in preds]
                out = mod(*inputs)

            cache[node_id] = out

        # If output node was never reached in the loop (edge case), try returning last cache value
        raise RuntimeError("Output node was not reached during forward pass")


# ── Topological sort ────────────────────────────────────────────

def _topo_sort(node_ids: list[str], in_edges: dict[str, list[str]]) -> list[str]:
    """Kahn's algorithm for topological ordering."""
    from collections import deque

    # Build out-edges and in-degree
    out_edges: dict[str, list[str]] = {nid: [] for nid in node_ids}
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}

    for nid in node_ids:
        for pred in in_edges.get(nid, []):
            out_edges[pred].append(nid)
            in_degree[nid] += 1

    queue: deque[str] = deque(nid for nid in node_ids if in_degree[nid] == 0)
    order: list[str] = []

    while queue:
        n = queue.popleft()
        order.append(n)
        for succ in out_edges[n]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(order) != len(node_ids):
        raise ValueError("GraphModel spec contains a cycle — DAG required")

    return order


# ── Public builder ───────────────────────────────────────────────

def build_graph_model(spec: dict[str, Any]) -> GraphModelModule:
    """
    Parse a v2 spec dict and return a ``GraphModelModule``.

    Raises
    ------
    ValueError
        If the spec is missing required fields, contains unknown layer types,
        contains a cycle, or is missing Input/Output nodes.
    """
    if spec.get("version") != 2:
        raise ValueError(f"build_graph_model requires version=2 spec, got version={spec.get('version')!r}")

    raw_nodes: list[dict] = spec.get("nodes", [])
    raw_edges: list[dict] = spec.get("edges", [])

    if not raw_nodes:
        raise ValueError("GraphModel spec has no nodes")

    # Identify Input / Output sentinel nodes
    input_nodes = [n for n in raw_nodes if n.get("type") == "Input"]
    output_nodes = [n for n in raw_nodes if n.get("type") == "Output"]

    if len(input_nodes) != 1:
        raise ValueError(f"GraphModel spec must have exactly 1 Input node, found {len(input_nodes)}")
    if len(output_nodes) != 1:
        raise ValueError(f"GraphModel spec must have exactly 1 Output node, found {len(output_nodes)}")

    input_id = input_nodes[0]["id"]
    output_id = output_nodes[0]["id"]

    node_ids = [n["id"] for n in raw_nodes]

    # Build in_edges: for each node, which nodes feed into it (ordered by edge definition)
    in_edges: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for edge in raw_edges:
        src = edge["source"]
        tgt = edge["target"]
        if src not in in_edges:
            raise ValueError(f"Edge references unknown source node '{src}'")
        if tgt not in in_edges:
            raise ValueError(f"Edge references unknown target node '{tgt}'")
        # Handle Input node's port — the actual tensor comes from the Input node itself
        in_edges[tgt].append(src)

    # Topological sort
    node_order = _topo_sort(node_ids, in_edges)

    # Build modules dict (skip Input / Output sentinels)
    modules_dict: dict[str, nn.Module] = {}
    for node_cfg in raw_nodes:
        mod = _build_node_module(node_cfg)
        if mod is not None:
            modules_dict[node_cfg["id"]] = mod

    total_params = sum(p.numel() for mod in modules_dict.values() for p in mod.parameters())
    logger.info(
        "Built GraphModelModule: %d nodes, %d edges, %s parameters",
        len(node_ids),
        len(raw_edges),
        f"{total_params:,}",
    )

    return GraphModelModule(
        node_order=node_order,
        modules_dict=modules_dict,
        in_edges=in_edges,
        input_node_id=input_id,
        output_node_id=output_id,
    )
