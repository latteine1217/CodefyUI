import { describe, it, expect } from 'vitest';
import { findEntryPoints } from './findEntryPoints';
import type { Node, Edge } from '@xyflow/react';

const node = (id: string, opts: { type?: string; isEntry?: boolean } = {}): Node => ({
  id,
  position: { x: 0, y: 0 },
  data: { id, type: opts.type ?? 'Dataset', isEntryPoint: opts.isEntry ?? false },
  type: 'baseNode',
});

const edge = (id: string, src: string, tgt: string, type: 'data' | 'trigger' = 'data'): Edge => ({
  id,
  source: src,
  target: tgt,
  data: { type },
});

describe('findEntryPoints', () => {
  it('returns explicit entry-pointed nodes', () => {
    const nodes = [node('a', { isEntry: true }), node('b')];
    expect(findEntryPoints(nodes, [])).toEqual(['a']);
  });

  it('returns Start nodes by type', () => {
    const nodes = [node('s', { type: 'Start' }), node('b')];
    expect(findEntryPoints(nodes, [])).toEqual(['s']);
  });

  it('returns nodes with incoming trigger edge', () => {
    const nodes = [node('s', { type: 'Start' }), node('ds')];
    const edges = [edge('e1', 's', 'ds', 'trigger')];
    const result = new Set(findEntryPoints(nodes, edges));
    expect(result).toEqual(new Set(['s', 'ds']));
  });

  it('returns empty when nothing is marked', () => {
    const nodes = [node('a'), node('b')];
    const edges = [edge('e1', 'a', 'b', 'data')];
    expect(findEntryPoints(nodes, edges)).toEqual([]);
  });
});
