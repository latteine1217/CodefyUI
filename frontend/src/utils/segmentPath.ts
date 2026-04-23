import type { Node, Edge } from '@xyflow/react';

/**
 * Return the set of nodes that lie on any path from ``headId`` to ``tailId``
 * via data edges.
 *
 * Used by the Teaching Inspector's Segment Compare feature to determine
 * which nodes to visually wrap with the orange bubble. If the tail is not
 * reachable from the head, returns an empty set.
 */
export function computeSegmentNodes(
  headId: string,
  tailId: string,
  _nodes: Node[],
  edges: Edge[],
): Set<string> {
  if (headId === tailId) return new Set([headId]);

  // Data adjacency: forward and reverse
  const forward = new Map<string, string[]>();
  const backward = new Map<string, string[]>();
  for (const e of edges) {
    // Skip trigger edges
    if (e.type === 'triggerEdge' || (e.data as any)?.type === 'trigger') continue;
    if (!forward.has(e.source)) forward.set(e.source, []);
    forward.get(e.source)!.push(e.target);
    if (!backward.has(e.target)) backward.set(e.target, []);
    backward.get(e.target)!.push(e.source);
  }

  const reachableFromHead = bfs(headId, forward);
  const reachesTail = bfs(tailId, backward);

  const segment = new Set<string>();
  for (const id of reachableFromHead) {
    if (reachesTail.has(id)) segment.add(id);
  }
  if (!segment.has(headId) || !segment.has(tailId)) return new Set();
  return segment;
}

function bfs(start: string, adj: Map<string, string[]>): Set<string> {
  const seen = new Set<string>([start]);
  const queue: string[] = [start];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    for (const next of adj.get(cur) ?? []) {
      if (!seen.has(next)) {
        seen.add(next);
        queue.push(next);
      }
    }
  }
  return seen;
}
