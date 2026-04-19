import { describe, it, expect } from 'vitest';
import { autoLayout } from './autoLayout';
import type { Node, Edge } from '@xyflow/react';

function makeStartNode(id: string, x = 0, y = 0): Node {
  return {
    id,
    position: { x, y },
    data: { id, type: 'Start' },
    type: 'start',
    width: 80,
    height: 40,
  };
}

function makeNode(id: string, x = 0, y = 0): Node {
  return {
    id,
    position: { x, y },
    data: { id, type: 'Dataset' },
    type: 'baseNode',
    width: 200,
    height: 80,
  };
}

function makeEdge(id: string, source: string, target: string, type: 'data' | 'trigger' = 'data'): Edge {
  return { id, source, target, data: { type } };
}

describe('autoLayout', () => {
  it('linear chain: A→B→C→D produces strictly increasing X at same Y', () => {
    const nodes = [
      makeStartNode('s'),
      makeNode('A'),
      makeNode('B'),
      makeNode('C'),
      makeNode('D'),
    ];
    const edges = [
      makeEdge('et', 's', 'A', 'trigger'),
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'B', 'C'),
      makeEdge('e3', 'C', 'D'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const sorted = ['A', 'B', 'C', 'D'].map((id) => result.find((n) => n.id === id)!);
    // Strictly increasing X
    expect(sorted[0].position.x).toBeLessThan(sorted[1].position.x);
    expect(sorted[1].position.x).toBeLessThan(sorted[2].position.x);
    expect(sorted[2].position.x).toBeLessThan(sorted[3].position.x);
    // Same Y (within rounding)
    const ys = sorted.map((n) => n.position.y);
    expect(Math.max(...ys) - Math.min(...ys)).toBeLessThan(5);
  });

  it('diamond: A→B,A→C,B→D,C→D — B and C stack vertically at same X', () => {
    const nodes = ['A', 'B', 'C', 'D'].map((id) => makeNode(id));
    const edges = [
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'A', 'C'),
      makeEdge('e3', 'B', 'D'),
      makeEdge('e4', 'C', 'D'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const B = result.find((n) => n.id === 'B')!;
    const C = result.find((n) => n.id === 'C')!;
    // B and C at the same X
    expect(Math.abs(B.position.x - C.position.x)).toBeLessThan(5);
    // Different Y
    expect(B.position.y).not.toBe(C.position.y);
  });

  it('two disconnected components → distinct Y bands', () => {
    const nodes = [
      makeStartNode('s1'),
      makeNode('A1'),
      makeNode('A2'),
      makeStartNode('s2'),
      makeNode('B1'),
      makeNode('B2'),
    ];
    const edges = [
      makeEdge('et1', 's1', 'A1', 'trigger'),
      makeEdge('e1', 'A1', 'A2'),
      makeEdge('et2', 's2', 'B1', 'trigger'),
      makeEdge('e2', 'B1', 'B2'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    const A1y = result.find((n) => n.id === 'A1')!.position.y;
    const B1y = result.find((n) => n.id === 'B1')!.position.y;
    expect(A1y).not.toBe(B1y);
  });

  it('cycle A→B→C→A does not crash and produces valid coordinates', () => {
    const nodes = ['A', 'B', 'C'].map((id) => makeNode(id));
    const edges = [
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'B', 'C'),
      makeEdge('e3', 'C', 'A'),
    ];
    const result = autoLayout(nodes, edges, 'all');
    for (const n of result) {
      expect(Number.isFinite(n.position.x)).toBe(true);
      expect(Number.isFinite(n.position.y)).toBe(true);
    }
  });

  it('mode=experiments leaves draft components untouched', () => {
    const nodes = [
      makeStartNode('s'),
      makeNode('live1', 100, 100),
      makeNode('live2', 200, 100),
      makeNode('draft1', 500, 500),
      makeNode('draft2', 700, 500),
    ];
    const edges = [
      makeEdge('et', 's', 'live1', 'trigger'),
      makeEdge('e1', 'live1', 'live2'),
      makeEdge('e2', 'draft1', 'draft2'),
    ];
    const result = autoLayout(nodes, edges, 'experiments');
    const draft1 = result.find((n) => n.id === 'draft1')!;
    const draft2 = result.find((n) => n.id === 'draft2')!;
    expect(draft1.position).toEqual({ x: 500, y: 500 });
    expect(draft2.position).toEqual({ x: 700, y: 500 });
  });

  it('mode=selected only moves selected nodes and preserves centroid', () => {
    const nodes = [
      makeNode('A', 100, 100),
      makeNode('B', 200, 100),
      makeNode('C', 300, 100),
      makeNode('untouched', 999, 999),
    ];
    const edges = [
      makeEdge('e1', 'A', 'B'),
      makeEdge('e2', 'B', 'C'),
    ];
    const selected = new Set(['A', 'B', 'C']);
    const result = autoLayout(nodes, edges, 'selected', selected);
    const untouched = result.find((n) => n.id === 'untouched')!;
    expect(untouched.position).toEqual({ x: 999, y: 999 });
    // Centroid of selection should be roughly preserved
    const beforeCentroid = { x: 200, y: 100 }; // (100+200+300)/3, 100
    const movedNodes = result.filter((n) => selected.has(n.id));
    const afterCentroid = {
      x: movedNodes.reduce((s, n) => s + n.position.x, 0) / movedNodes.length,
      y: movedNodes.reduce((s, n) => s + n.position.y, 0) / movedNodes.length,
    };
    expect(Math.abs(afterCentroid.x - beforeCentroid.x)).toBeLessThan(50);
    expect(Math.abs(afterCentroid.y - beforeCentroid.y)).toBeLessThan(50);
  });
});
