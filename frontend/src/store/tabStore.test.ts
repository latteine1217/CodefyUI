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
