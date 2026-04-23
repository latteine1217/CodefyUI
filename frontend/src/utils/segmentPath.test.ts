import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import { computeSegmentNodes } from './segmentPath';

function mkNode(id: string): Node {
  return { id, type: 'baseNode', position: { x: 0, y: 0 }, data: { id } as any };
}

function mkEdge(id: string, source: string, target: string): Edge {
  return { id, source, target };
}

describe('computeSegmentNodes', () => {
  it('linear chain includes all nodes between head and tail', () => {
    const nodes = ['a', 'b', 'c', 'd'].map(mkNode);
    const edges = [
      mkEdge('e1', 'a', 'b'),
      mkEdge('e2', 'b', 'c'),
      mkEdge('e3', 'c', 'd'),
    ];
    const seg = computeSegmentNodes('a', 'd', nodes, edges);
    expect(seg).toEqual(new Set(['a', 'b', 'c', 'd']));
  });

  it('branch that rejoins includes both branches', () => {
    const nodes = ['a', 'b1', 'b2', 'c'].map(mkNode);
    const edges = [
      mkEdge('e1', 'a', 'b1'),
      mkEdge('e2', 'a', 'b2'),
      mkEdge('e3', 'b1', 'c'),
      mkEdge('e4', 'b2', 'c'),
    ];
    const seg = computeSegmentNodes('a', 'c', nodes, edges);
    expect(seg).toEqual(new Set(['a', 'b1', 'b2', 'c']));
  });

  it('excludes sibling branches that do not reach tail', () => {
    const nodes = ['a', 'b', 'c', 'dead'].map(mkNode);
    const edges = [
      mkEdge('e1', 'a', 'b'),
      mkEdge('e2', 'b', 'c'),
      mkEdge('e3', 'b', 'dead'),
    ];
    const seg = computeSegmentNodes('a', 'c', nodes, edges);
    expect(seg).toEqual(new Set(['a', 'b', 'c']));
  });

  it('disconnected head/tail returns empty set', () => {
    const nodes = ['a', 'b', 'c'].map(mkNode);
    const edges = [mkEdge('e1', 'a', 'b')];
    const seg = computeSegmentNodes('a', 'c', nodes, edges);
    expect(seg).toEqual(new Set());
  });

  it('head === tail returns that one node', () => {
    const seg = computeSegmentNodes('a', 'a', [mkNode('a')], []);
    expect(seg).toEqual(new Set(['a']));
  });

  it('ignores trigger edges', () => {
    const nodes = ['start', 'a', 'b'].map(mkNode);
    const edges: Edge[] = [
      {
        id: 'et',
        source: 'start',
        target: 'a',
        type: 'triggerEdge',
        data: { type: 'trigger' },
      },
      mkEdge('e1', 'a', 'b'),
    ];
    const seg = computeSegmentNodes('start', 'b', nodes, edges);
    expect(seg).toEqual(new Set());
  });
});
