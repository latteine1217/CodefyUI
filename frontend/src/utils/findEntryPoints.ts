import type { Node, Edge } from '@xyflow/react';

/**
 * Return ids of nodes that are entry points.
 *
 * A node is an entry point if it has at least one incoming trigger edge
 * (i.e. it is connected from a Start node). Start nodes themselves are
 * NOT entry points — they are markers.
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
  return nodes.filter((n) => triggerTargets.has(n.id)).map((n) => n.id);
}
