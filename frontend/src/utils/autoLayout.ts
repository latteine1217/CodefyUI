import dagre from '@dagrejs/dagre';
import type { Node, Edge } from '@xyflow/react';

export type LayoutMode = 'experiments' | 'all' | 'selected';

const NODE_W = 200;
const NODE_H = 80;
const NODESEP = 40;
const RANKSEP = 80;
const LANE_GAP = 60;

function isEntryPointOrStart(node: Node): boolean {
  return node.type === 'start' || (node.data as any)?.type === 'Start';
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
    const w = node.measured?.width ?? node.width ?? NODE_W;
    const h = node.measured?.height ?? node.height ?? NODE_H;
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
    if (compNodes.some(isEntryPointOrStart)) {
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
      x: sel.reduce((s, n) => s + n.position.x + (n.measured?.width ?? n.width ?? NODE_W) / 2, 0) / sel.length,
      y: sel.reduce((s, n) => s + n.position.y + (n.measured?.height ?? n.height ?? NODE_H) / 2, 0) / sel.length,
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
      hasEntryPoint: compNodes.some(isEntryPointOrStart),
      bounds: { minY, maxY },
    };
  });

  let finalPositions = packIntoSwimLanes(laidOut);

  // Selected mode: shift result so the centroid matches the original
  if (mode === 'selected' && originalCentroid) {
    const sel = Array.from(finalPositions.entries());
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));
    const newCentroid = {
      x: sel.reduce((s, [id, p]) => s + p.x + (nodeMap.get(id)?.measured?.width ?? nodeMap.get(id)?.width ?? NODE_W) / 2, 0) / sel.length,
      y: sel.reduce((s, [id, p]) => s + p.y + (nodeMap.get(id)?.measured?.height ?? nodeMap.get(id)?.height ?? NODE_H) / 2, 0) / sel.length,
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
