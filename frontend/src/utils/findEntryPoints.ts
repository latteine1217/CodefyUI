import type { Node, Edge } from '@xyflow/react';

/**
 * Return ids of nodes that are entry points.
 *
 * A node is an entry point if any of:
 *   1. Its `isEntryPoint` field is true.
 *   2. It is of type "Start" (Start nodes are always entry points).
 *   3. It has at least one incoming edge of type "trigger".
 *
 * The order of returned ids matches the order in `nodes` for determinism.
 *
 * Mirrors the backend `find_entry_points` in
 * `backend/app/core/graph_engine.py`.
 */
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
