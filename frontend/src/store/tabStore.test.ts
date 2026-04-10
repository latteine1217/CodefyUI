import { describe, it, expect, beforeEach } from 'vitest';
import { useTabStore } from './tabStore';

describe('toggleEntryPoint', () => {
  beforeEach(() => {
    // Reset store to known state
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    useTabStore.getState().addTab('test');
  });

  it('sets isEntryPoint to true on a node that was unmarked', () => {
    const tabId = useTabStore.getState().activeTabId!;
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((t) =>
        t.id === tabId
          ? {
              ...t,
              nodes: [
                {
                  id: 'n1',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  data: { id: 'n1', type: 'Dataset', isEntryPoint: false },
                },
              ] as any,
            }
          : t,
      ),
    }));

    useTabStore.getState().toggleEntryPoint('n1');
    const tab = useTabStore.getState().tabs.find((t) => t.id === tabId)!;
    expect(tab.nodes[0].data.isEntryPoint).toBe(true);
  });

  it('clears isEntryPoint on a node that was marked', () => {
    const tabId = useTabStore.getState().activeTabId!;
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((t) =>
        t.id === tabId
          ? {
              ...t,
              nodes: [
                {
                  id: 'n1',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  data: { id: 'n1', type: 'Dataset', isEntryPoint: true },
                },
              ] as any,
            }
          : t,
      ),
    }));

    useTabStore.getState().toggleEntryPoint('n1');
    const tab = useTabStore.getState().tabs.find((t) => t.id === tabId)!;
    expect(tab.nodes[0].data.isEntryPoint).toBe(false);
  });
});

describe('applyLayout', () => {
  beforeEach(() => {
    // Reset store to known state
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    useTabStore.getState().addTab('test');
  });

  it('repositions nodes in an entry-pointed component for experiments mode', () => {
    const tabId = useTabStore.getState().activeTabId!;
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((t) =>
        t.id === tabId
          ? {
              ...t,
              nodes: [
                {
                  id: 'a',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  width: 200,
                  height: 80,
                  data: { id: 'a', type: 'Dataset', isEntryPoint: true },
                },
                {
                  id: 'b',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  width: 200,
                  height: 80,
                  data: { id: 'b', type: 'Dataset', isEntryPoint: false },
                },
                {
                  id: 'c',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  width: 200,
                  height: 80,
                  data: { id: 'c', type: 'Dataset', isEntryPoint: false },
                },
              ] as any,
              edges: [
                { id: 'e1', source: 'a', target: 'b' },
                { id: 'e2', source: 'b', target: 'c' },
              ] as any,
            }
          : t,
      ),
    }));

    useTabStore.getState().applyLayout('experiments');
    const tab = useTabStore.getState().tabs.find((t) => t.id === tabId)!;
    const a = tab.nodes.find((n) => n.id === 'a')!;
    const b = tab.nodes.find((n) => n.id === 'b')!;
    const c = tab.nodes.find((n) => n.id === 'c')!;

    // Dagre LR layout: A → B → C should produce strictly increasing X
    expect(a.position.x).toBeLessThan(b.position.x);
    expect(b.position.x).toBeLessThan(c.position.x);

    // At least one node must have been moved from its original (0, 0) position
    expect(a.position.x !== 0 || b.position.x !== 0 || c.position.x !== 0).toBe(true);

    // Undo snapshot was pushed so Ctrl+Z reverts the layout
    expect(tab.undoStack.length).toBe(1);
  });
});
