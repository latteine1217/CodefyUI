import { describe, it, expect } from 'vitest';
import { findEntryPoints } from './findEntryPoints';
import type { Node, Edge } from '@xyflow/react';

const node = (id: string, opts: { type?: string } = {}): Node => ({
  id,
  position: { x: 0, y: 0 },
  data: { id, type: opts.type ?? 'Dataset' },
  type: 'baseNode',
});

const edge = (id: string, src: string, tgt: string, type: 'data' | 'trigger' = 'data'): Edge => ({
  id,
  source: src,
  target: tgt,
  data: { type },
});

describe('findEntryPoints', () => {
  it('returns nodes with incoming trigger edge', () => {
    const nodes = [node('s', { type: 'Start' }), node('ds')];
    const edges = [edge('e1', 's', 'ds', 'trigger')];
    expect(findEntryPoints(nodes, edges)).toEqual(['ds']);
  });

  it('does NOT include Start node itself as entry point', () => {
    const nodes = [node('s', { type: 'Start' }), node('ds')];
    const edges = [edge('e1', 's', 'ds', 'trigger')];
    const result = findEntryPoints(nodes, edges);
    expect(result).not.toContain('s');
  });

  it('returns multiple trigger targets from one Start', () => {
    const nodes = [node('s', { type: 'Start' }), node('a'), node('b')];
    const edges = [
      edge('e1', 's', 'a', 'trigger'),
      edge('e2', 's', 'b', 'trigger'),
    ];
    expect(new Set(findEntryPoints(nodes, edges))).toEqual(new Set(['a', 'b']));
  });

  it('returns empty when nothing has a trigger edge', () => {
    const nodes = [node('a'), node('b')];
    const edges = [edge('e1', 'a', 'b', 'data')];
    expect(findEntryPoints(nodes, edges)).toEqual([]);
  });
});
