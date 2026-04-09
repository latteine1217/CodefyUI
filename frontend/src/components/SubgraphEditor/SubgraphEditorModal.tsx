import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  BackgroundVariant,
  useReactFlow,
  applyNodeChanges,
  applyEdgeChanges,
  type Node,
  type Edge,
  type NodeChange,
  type EdgeChange,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';
import { generateId } from '../../utils';
import { LayerNode } from './LayerNode';
import type { ParamDefinition } from '../../types';

// ── Layer definitions (matches backend _build_layer) ──

interface LayerDef {
  type: string;
  category: string;
  color: string;
  params: ParamDefinition[];
}

const p_int = (name: string, def: number, desc: string, min: number | null = 1): ParamDefinition => (
  { name, param_type: 'int', default: def, description: desc, options: [], min_value: min, max_value: null }
);
const p_float = (name: string, def: number, desc: string, min: number | null = 0, max: number | null = null): ParamDefinition => (
  { name, param_type: 'float', default: def, description: desc, options: [], min_value: min, max_value: max }
);

const LAYER_DEFS: LayerDef[] = [
  // Convolution
  {
    type: 'Conv2d', category: 'Convolution', color: '#4CAF50',
    params: [p_int('in_channels', 1, 'Input channels'), p_int('out_channels', 32, 'Output channels'), p_int('kernel_size', 3, 'Kernel size'), p_int('stride', 1, 'Stride'), p_int('padding', 1, 'Padding', 0)],
  },
  {
    type: 'Conv1d', category: 'Convolution', color: '#4CAF50',
    params: [p_int('in_channels', 1, 'Input channels'), p_int('out_channels', 32, 'Output channels'), p_int('kernel_size', 3, 'Kernel size'), p_int('stride', 1, 'Stride'), p_int('padding', 1, 'Padding', 0)],
  },
  {
    type: 'ConvTranspose2d', category: 'Convolution', color: '#4CAF50',
    params: [p_int('in_channels', 32, 'Input channels'), p_int('out_channels', 16, 'Output channels'), p_int('kernel_size', 3, 'Kernel size'), p_int('stride', 1, 'Stride'), p_int('padding', 1, 'Padding', 0)],
  },
  // Normalization
  {
    type: 'BatchNorm2d', category: 'Normalization', color: '#9C27B0',
    params: [p_int('num_features', 32, 'Number of features')],
  },
  {
    type: 'BatchNorm1d', category: 'Normalization', color: '#9C27B0',
    params: [p_int('num_features', 32, 'Number of features')],
  },
  {
    type: 'LayerNorm', category: 'Normalization', color: '#9C27B0',
    params: [p_int('normalized_shape', 512, 'Normalized shape')],
  },
  {
    type: 'GroupNorm', category: 'Normalization', color: '#9C27B0',
    params: [p_int('num_groups', 8, 'Number of groups'), p_int('num_channels', 32, 'Number of channels')],
  },
  {
    type: 'InstanceNorm2d', category: 'Normalization', color: '#9C27B0',
    params: [p_int('num_features', 32, 'Number of features')],
  },
  // Pooling
  {
    type: 'MaxPool2d', category: 'Pooling', color: '#2196F3',
    params: [p_int('kernel_size', 2, 'Kernel size'), p_int('stride', 2, 'Stride')],
  },
  {
    type: 'AvgPool2d', category: 'Pooling', color: '#2196F3',
    params: [p_int('kernel_size', 2, 'Kernel size'), p_int('stride', 2, 'Stride')],
  },
  {
    type: 'AdaptiveAvgPool2d', category: 'Pooling', color: '#2196F3',
    params: [p_int('output_size', 1, 'Output size')],
  },
  // Regularization
  {
    type: 'Dropout', category: 'Regularization', color: '#FF9800',
    params: [p_float('p', 0.5, 'Dropout probability', 0, 1)],
  },
  // Linear
  {
    type: 'Linear', category: 'Linear', color: '#00BCD4',
    params: [p_int('in_features', 512, 'Input features'), p_int('out_features', 10, 'Output features')],
  },
  {
    type: 'Embedding', category: 'Linear', color: '#00BCD4',
    params: [p_int('num_embeddings', 10000, 'Vocabulary size'), p_int('embedding_dim', 256, 'Embedding dimension')],
  },
  // Utility
  {
    type: 'Flatten', category: 'Utility', color: '#607D8B',
    params: [],
  },
  // Activations
  { type: 'ReLU', category: 'Activation', color: '#F44336', params: [] },
  { type: 'LeakyReLU', category: 'Activation', color: '#F44336', params: [] },
  { type: 'GELU', category: 'Activation', color: '#F44336', params: [] },
  { type: 'SiLU', category: 'Activation', color: '#F44336', params: [] },
  { type: 'Mish', category: 'Activation', color: '#F44336', params: [] },
  { type: 'ELU', category: 'Activation', color: '#F44336', params: [] },
  { type: 'SELU', category: 'Activation', color: '#F44336', params: [] },
  { type: 'PReLU', category: 'Activation', color: '#F44336', params: [] },
  { type: 'Sigmoid', category: 'Activation', color: '#F44336', params: [] },
  { type: 'Tanh', category: 'Activation', color: '#F44336', params: [] },
  { type: 'Hardswish', category: 'Activation', color: '#F44336', params: [] },
  { type: 'Softmax', category: 'Activation', color: '#F44336', params: [] },
];

const LAYER_DEF_MAP = new Map(LAYER_DEFS.map((d) => [d.type, d]));

// ── Types ──

interface LayerNodeData {
  layerType: string;
  params: Record<string, any>;
  color: string;
  [key: string]: unknown;
}

// ── Convert layers JSON to nodes/edges ──

function layersToFlow(layersJson: string): { nodes: Node<LayerNodeData>[]; edges: Edge[] } {
  let layers: Record<string, any>[];
  try {
    layers = JSON.parse(layersJson);
  } catch {
    layers = [];
  }
  if (!Array.isArray(layers)) layers = [];

  const nodes: Node<LayerNodeData>[] = [];
  const edges: Edge[] = [];

  layers.forEach((layer, i) => {
    const type = layer.type ?? 'Unknown';
    const def = LAYER_DEF_MAP.get(type);
    const params = { ...layer };
    delete params.type;

    const id = generateId();
    nodes.push({
      id,
      type: 'layerNode',
      position: { x: 200, y: i * 100 },
      data: {
        layerType: type,
        params,
        color: def?.color ?? '#607D8B',
      },
    });

    if (i > 0) {
      edges.push({
        id: generateId(),
        source: nodes[i - 1].id,
        target: id,
        style: { stroke: '#555', strokeWidth: 2 },
      });
    }
  });

  return { nodes, edges };
}

// ── Convert nodes/edges back to layers JSON ──

function flowToLayersJson(nodes: Node<LayerNodeData>[], edges: Edge[]): string {
  // Build adjacency to determine order
  const outgoing = new Map<string, string>();
  for (const e of edges) {
    outgoing.set(e.source, e.target);
  }

  // Find the head node (not a target of any edge)
  const targets = new Set(edges.map((e) => e.target));
  const heads = nodes.filter((n) => !targets.has(n.id));

  // Walk the chain from head
  const ordered: Node<LayerNodeData>[] = [];
  const visited = new Set<string>();

  // If there are multiple disconnected chains or no edges, fall back to vertical position order
  if (heads.length !== 1 || edges.length !== nodes.length - 1) {
    // Sort by Y position
    const sorted = [...nodes].sort((a, b) => a.position.y - b.position.y);
    for (const n of sorted) {
      ordered.push(n);
    }
  } else {
    let current: string | undefined = heads[0].id;
    while (current && !visited.has(current)) {
      visited.add(current);
      const node = nodes.find((n) => n.id === current);
      if (node) ordered.push(node);
      current = outgoing.get(current);
    }
    // Add any unvisited nodes (disconnected)
    for (const n of nodes) {
      if (!visited.has(n.id)) ordered.push(n);
    }
  }

  const layers = ordered.map((n) => ({
    type: n.data.layerType,
    ...n.data.params,
  }));

  return JSON.stringify(layers);
}

// ── Layer Palette Item ──

function LayerPaletteItem({ def }: { def: LayerDef }) {
  const [hovered, setHovered] = useState(false);

  const handleDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData('application/subgraph-layer', def.type);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        padding: '6px 10px',
        margin: '2px 0',
        borderRadius: 5,
        cursor: 'grab',
        background: hovered ? '#2a2a2a' : 'transparent',
        border: '1px solid',
        borderColor: hovered ? '#444' : 'transparent',
        transition: 'all 0.15s',
        userSelect: 'none',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: def.color,
          flexShrink: 0,
        }}
      />
      <span style={{ fontSize: '0.8125rem', color: '#ddd', fontWeight: 500 }}>
        {def.type}
      </span>
      {def.params.length > 0 && (
        <span style={{ fontSize: '0.625rem', color: '#666', marginLeft: 'auto' }}>
          {def.params.length}p
        </span>
      )}
    </div>
  );
}

// ── Param Editor Panel ──

function ParamEditor({
  node,
  onParamChange,
  onDelete,
}: {
  node: Node<LayerNodeData>;
  onParamChange: (nodeId: string, paramName: string, value: any) => void;
  onDelete: (nodeId: string) => void;
}) {
  const { t } = useI18n();
  const def = LAYER_DEF_MAP.get(node.data.layerType);

  return (
    <div style={{ padding: '12px 10px' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          marginBottom: 12,
          paddingBottom: 8,
          borderBottom: '1px solid #333',
        }}
      >
        <span
          style={{
            width: 10,
            height: 10,
            borderRadius: '50%',
            background: node.data.color,
            flexShrink: 0,
          }}
        />
        <span style={{ fontSize: '0.9375rem', fontWeight: 700, color: '#eee', flex: 1 }}>
          {node.data.layerType}
        </span>
        <button
          onClick={() => onDelete(node.id)}
          style={{
            padding: '3px 8px',
            background: '#3a1515',
            border: '1px solid #F44336',
            borderRadius: 4,
            color: '#F44336',
            fontSize: '0.6875rem',
            cursor: 'pointer',
          }}
        >
          {t('subgraph.deleteLayer')}
        </button>
      </div>

      {(!def || def.params.length === 0) && (
        <div style={{ fontSize: '0.75rem', color: '#666', textAlign: 'center', padding: '8px 0' }}>
          {t('subgraph.noParams')}
        </div>
      )}

      {def?.params.map((p) => {
        const val = node.data.params[p.name] ?? p.default;
        return (
          <div key={p.name} style={{ marginBottom: 8 }}>
            <label style={{ display: 'block', fontSize: '0.75rem', color: '#999', marginBottom: 3, fontWeight: 500 }}>
              {p.name}
            </label>
            <input
              type={p.param_type === 'float' ? 'number' : p.param_type === 'int' ? 'number' : 'text'}
              value={val}
              step={p.param_type === 'float' ? 'any' : 1}
              min={p.min_value ?? undefined}
              max={p.max_value ?? undefined}
              onChange={(e) => {
                let v: any = e.target.value;
                if (p.param_type === 'int') v = parseInt(v, 10);
                else if (p.param_type === 'float') v = parseFloat(v);
                onParamChange(node.id, p.name, v);
              }}
              style={{
                width: '100%',
                padding: '5px 8px',
                background: '#222',
                border: '1px solid #444',
                borderRadius: 4,
                color: '#ddd',
                fontSize: '0.8125rem',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </div>
        );
      })}
    </div>
  );
}

// ── Inner Flow (needs ReactFlowProvider wrapping) ──

const nodeTypes: NodeTypes = {
  layerNode: LayerNode,
};

function SubgraphFlowInner({
  initialLayersJson,
  onApply,
  onCancel,
}: {
  initialLayersJson: string;
  onApply: (layersJson: string) => void;
  onCancel: () => void;
}) {
  const { t } = useI18n();
  const { screenToFlowPosition, fitView } = useReactFlow();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const initial = useMemo(() => layersToFlow(initialLayersJson), [initialLayersJson]);
  const [nodes, setNodes] = useState<Node<LayerNodeData>[]>(initial.nodes);
  const [edges, setEdges] = useState<Edge[]>(initial.edges);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [snapEnabled, setSnapEnabled] = useState(false);

  useEffect(() => {
    setTimeout(() => fitView({ padding: 0.3 }), 50);
  }, [fitView]);

  const selectedNode = nodes.find((n) => n.id === selectedNodeId) ?? null;

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setNodes((nds) => applyNodeChanges(changes, nds) as Node<LayerNodeData>[]);
    },
    []
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setEdges((eds) => applyEdgeChanges(changes, eds));
    },
    []
  );

  // Auto-connect: rebuild chain by Y position after drag
  const rebuildChain = useCallback((currentNodes: Node<LayerNodeData>[]) => {
    const sorted = [...currentNodes].sort((a, b) => a.position.y - b.position.y);
    const newEdges: Edge[] = [];
    for (let i = 1; i < sorted.length; i++) {
      newEdges.push({
        id: `chain-${sorted[i - 1].id}-${sorted[i].id}`,
        source: sorted[i - 1].id,
        target: sorted[i].id,
        style: { stroke: '#555', strokeWidth: 2 },
      });
    }
    setEdges(newEdges);
  }, []);

  const handleNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setNodes((prevNodes) => {
        const updated = applyNodeChanges(changes, prevNodes) as Node<LayerNodeData>[];
        // Rebuild chain if any node was dragged
        const hasDrag = changes.some((c) => c.type === 'position' && 'dragging' in c && !c.dragging);
        if (hasDrag) {
          setTimeout(() => rebuildChain(updated), 0);
        }
        return updated;
      });
    },
    [rebuildChain]
  );

  const addLayer = useCallback(
    (layerType: string, position: { x: number; y: number }) => {
      const def = LAYER_DEF_MAP.get(layerType);
      if (!def) return;

      const defaultParams: Record<string, any> = {};
      for (const p of def.params) {
        defaultParams[p.name] = p.default;
      }

      const newNode: Node<LayerNodeData> = {
        id: generateId(),
        type: 'layerNode',
        position,
        data: {
          layerType,
          params: defaultParams,
          color: def.color,
        },
      };

      setNodes((prev) => {
        const updated = [...prev, newNode];
        setTimeout(() => rebuildChain(updated), 0);
        return updated;
      });
    },
    [rebuildChain]
  );

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const layerType = event.dataTransfer.getData('application/subgraph-layer');
      if (!layerType) return;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      addLayer(layerType, position);
    },
    [screenToFlowPosition, addLayer]
  );

  const handleParamChange = useCallback((nodeId: string, paramName: string, value: any) => {
    setNodes((prev) =>
      prev.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, params: { ...n.data.params, [paramName]: value } } }
          : n
      )
    );
  }, []);

  const handleDeleteLayer = useCallback((nodeId: string) => {
    setNodes((prev) => {
      const updated = prev.filter((n) => n.id !== nodeId);
      setTimeout(() => rebuildChain(updated), 0);
      return updated;
    });
    setSelectedNodeId((prev) => (prev === nodeId ? null : prev));
  }, [rebuildChain]);

  const handleApply = () => {
    onApply(flowToLayersJson(nodes, edges));
  };

  const handleExport = () => {
    const json = flowToLayersJson(nodes, edges);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model_architecture.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const parsed = JSON.parse(text);
        if (!Array.isArray(parsed)) throw new Error('Expected JSON array');
        const { nodes: newNodes, edges: newEdges } = layersToFlow(text);
        setNodes(newNodes);
        setEdges(newEdges);
        setSelectedNodeId(null);
        setTimeout(() => fitView({ padding: 0.3 }), 50);
      } catch (err) {
        useToastStore.getState().addToast(t('subgraph.import.fail', { error: String(err) }), 'error');
      }
    };
    reader.readAsText(file);
    // Reset input so the same file can be re-selected
    event.target.value = '';
  };

  // Filter layer palette
  const filteredDefs = search.trim()
    ? LAYER_DEFS.filter((d) => d.type.toLowerCase().includes(search.toLowerCase()))
    : LAYER_DEFS;

  const groupedDefs = useMemo(() => {
    const groups: Record<string, LayerDef[]> = {};
    for (const d of filteredDefs) {
      if (!groups[d.category]) groups[d.category] = [];
      groups[d.category].push(d);
    }
    return groups;
  }, [filteredDefs]);

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(0,0,0,0.7)',
        backdropFilter: 'blur(6px)',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <div
        style={{
          background: '#1a1a1a',
          border: '1px solid #333',
          borderRadius: 12,
          width: '95vw',
          maxWidth: 1700,
          height: '90vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: '12px 20px',
            borderBottom: '2px solid #F44336',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexShrink: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: '1.0625rem', fontWeight: 700, color: '#eee' }}>
              {t('subgraph.title')}
            </span>
            <span
              style={{
                fontSize: '0.6875rem',
                background: 'rgba(244,67,54,0.15)',
                color: '#F44336',
                padding: '2px 8px',
                borderRadius: 3,
                fontWeight: 600,
              }}
            >
              {t('subgraph.layerCount', { count: nodes.length })}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <button
              onClick={() => setSnapEnabled((v) => !v)}
              title={t('subgraph.snapTitle')}
              style={{
                padding: '5px 12px',
                background: snapEnabled ? 'rgba(244,67,54,0.18)' : '#2a2a2a',
                border: snapEnabled ? '1px solid #F44336' : '1px solid #444',
                borderRadius: 5,
                color: snapEnabled ? '#F44336' : '#aaa',
                fontSize: '0.75rem',
                cursor: 'pointer',
                fontWeight: 600,
              }}
            >
              {snapEnabled ? t('subgraph.snapOn') : t('subgraph.snapOff')}
            </button>
            <button
              onClick={handleImport}
              title={t('subgraph.import.title')}
              style={{
                padding: '5px 12px',
                background: '#2a2a2a',
                border: '1px solid #444',
                borderRadius: 5,
                color: '#aaa',
                fontSize: '0.75rem',
                cursor: 'pointer',
              }}
            >
              {t('subgraph.import')}
            </button>
            <button
              onClick={handleExport}
              title={t('subgraph.export.title')}
              style={{
                padding: '5px 12px',
                background: '#2a2a2a',
                border: '1px solid #444',
                borderRadius: 5,
                color: '#aaa',
                fontSize: '0.75rem',
                cursor: 'pointer',
              }}
            >
              {t('subgraph.export')}
            </button>
            <button
              onClick={onCancel}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#666',
                fontSize: '1.125rem',
                cursor: 'pointer',
                padding: '0 4px',
                lineHeight: 1,
                marginLeft: 8,
              }}
            >
              ✕
            </button>
          </div>
        </div>

        {/* Body: palette | canvas | param editor */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* Layer Palette */}
          <div
            style={{
              width: 180,
              borderRight: '1px solid #2a2a2a',
              display: 'flex',
              flexDirection: 'column',
              flexShrink: 0,
            }}
          >
            <div style={{ padding: '8px 10px', borderBottom: '1px solid #2a2a2a', flexShrink: 0 }}>
              <div
                style={{
                  fontSize: '0.6875rem',
                  fontWeight: 700,
                  color: '#888',
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  marginBottom: 6,
                }}
              >
                {t('subgraph.palette')}
              </div>
              <input
                type="text"
                placeholder={t('subgraph.searchLayers')}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{
                  width: '100%',
                  padding: '4px 8px',
                  background: '#222',
                  border: '1px solid #333',
                  borderRadius: 4,
                  color: '#ddd',
                  fontSize: '0.75rem',
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
              />
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '4px 6px' }}>
              {Object.entries(groupedDefs).map(([category, defs]) => (
                <div key={category} style={{ marginBottom: 6 }}>
                  <div
                    style={{
                      fontSize: '0.625rem',
                      fontWeight: 700,
                      color: '#666',
                      letterSpacing: '0.06em',
                      textTransform: 'uppercase',
                      padding: '4px 4px 2px',
                    }}
                  >
                    {category}
                  </div>
                  {defs.map((d) => (
                    <LayerPaletteItem key={d.type} def={d} />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Canvas */}
          <div style={{ flex: 1, position: 'relative' }}>
            {nodes.length === 0 && (
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  zIndex: 10,
                  pointerEvents: 'none',
                }}
              >
                <div
                  style={{
                    color: '#555',
                    fontSize: '0.875rem',
                    textAlign: 'center',
                    padding: '20px',
                  }}
                >
                  {t('subgraph.empty')}
                </div>
              </div>
            )}
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={handleNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={(_, node) => setSelectedNodeId(node.id)}
              onPaneClick={() => setSelectedNodeId(null)}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              nodeTypes={nodeTypes}
              fitView
              snapToGrid={snapEnabled}
              snapGrid={[20, 20]}
              proOptions={{ hideAttribution: true }}
              deleteKeyCode="Delete"
              onNodesDelete={(deleted) => {
                const ids = new Set(deleted.map((n) => n.id));
                setNodes((prev) => {
                  const updated = prev.filter((n) => !ids.has(n.id));
                  setTimeout(() => rebuildChain(updated), 0);
                  return updated;
                });
                setSelectedNodeId((prev) => (prev && ids.has(prev) ? null : prev));
              }}
              style={{ background: '#111' }}
              defaultEdgeOptions={{
                animated: false,
                style: { stroke: '#555', strokeWidth: 2 },
              }}
            >
              <Background
                color="#2a2a2a"
                variant={BackgroundVariant.Dots}
                gap={20}
                size={1}
              />
              <Controls />
            </ReactFlow>
          </div>

          {/* Param Editor */}
          <div
            style={{
              width: 220,
              borderLeft: '1px solid #2a2a2a',
              display: 'flex',
              flexDirection: 'column',
              flexShrink: 0,
            }}
          >
            <div
              style={{
                padding: '10px 10px',
                borderBottom: '1px solid #2a2a2a',
                fontSize: '0.6875rem',
                fontWeight: 700,
                color: '#888',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                flexShrink: 0,
              }}
            >
              {t('subgraph.params')}
            </div>
            <div style={{ flex: 1, overflowY: 'auto' }}>
              {selectedNode ? (
                <ParamEditor
                  node={selectedNode}
                  onParamChange={handleParamChange}
                  onDelete={handleDeleteLayer}
                />
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
            </div>
          </div>
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '10px 20px',
            borderTop: '1px solid #2a2a2a',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 8,
            flexShrink: 0,
          }}
        >
          <button
            onClick={onCancel}
            style={{
              padding: '7px 16px',
              background: '#2a2a2a',
              border: '1px solid #444',
              borderRadius: 6,
              color: '#aaa',
              fontSize: '0.8125rem',
              cursor: 'pointer',
            }}
          >
            {t('subgraph.cancel')}
          </button>
          <button
            onClick={handleApply}
            style={{
              padding: '7px 16px',
              background: '#F44336',
              border: 'none',
              borderRadius: 6,
              color: '#fff',
              fontSize: '0.8125rem',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            {t('subgraph.apply')}
          </button>
        </div>

        {/* Hidden file input for import */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          style={{ display: 'none' }}
          onChange={handleFileSelect}
        />
      </div>
    </div>
  );
}

// ── Main Export ──

export function SubgraphEditorModal() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const closeSubgraphModal = useTabStore((s) => s.closeSubgraphModal);
  const updateSubgraphLayers = useTabStore((s) => s.updateSubgraphLayers);

  const nodeId = activeTab.subgraphModalNodeId;
  const node = activeTab.nodes.find((n) => n.id === nodeId);

  if (!nodeId || !node) return null;

  const layersJson = (node.data.params?.layers as string) ?? '[]';

  const handleApply = (newLayersJson: string) => {
    updateSubgraphLayers(nodeId, newLayersJson);
    closeSubgraphModal();
  };

  return (
    <ReactFlowProvider>
      <SubgraphFlowInner
        initialLayersJson={layersJson}
        onApply={handleApply}
        onCancel={closeSubgraphModal}
      />
    </ReactFlowProvider>
  );
}
