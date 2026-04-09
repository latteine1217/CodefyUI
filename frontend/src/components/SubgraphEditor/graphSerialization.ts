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
