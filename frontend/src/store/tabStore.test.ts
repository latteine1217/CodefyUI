import { describe, it, expect, beforeEach } from 'vitest';
import { useTabStore } from './tabStore';

describe('applyLayout', () => {
  beforeEach(() => {
    // Reset store to known state
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    useTabStore.getState().addTab('test');
  });

  it('repositions nodes in a component with a Start node for experiments mode', () => {
    const tabId = useTabStore.getState().activeTabId!;
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((t) =>
        t.id === tabId
          ? {
              ...t,
              nodes: [
                {
                  id: 's',
                  type: 'start',
                  position: { x: 0, y: 0 },
                  width: 80,
                  height: 40,
                  data: { id: 's', type: 'Start' },
                },
                {
                  id: 'a',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  width: 200,
                  height: 80,
                  data: { id: 'a', type: 'Dataset' },
                },
                {
                  id: 'b',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  width: 200,
                  height: 80,
                  data: { id: 'b', type: 'DataLoader' },
                },
                {
                  id: 'c',
                  type: 'baseNode',
                  position: { x: 0, y: 0 },
                  width: 200,
                  height: 80,
                  data: { id: 'c', type: 'Model' },
                },
              ] as any,
              edges: [
                { id: 'et', source: 's', target: 'a', data: { type: 'trigger' } },
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

describe('Teaching Inspector actions', () => {
  beforeEach(() => {
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    useTabStore.getState().addTab('test');
  });

  it('recordOutputs defaults to true', () => {
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.recordOutputs).toBe(true);
  });

  it('toggleRecord flips the flag', () => {
    useTabStore.getState().toggleRecord();
    expect(useTabStore.getState().getActiveTab().recordOutputs).toBe(false);
    useTabStore.getState().toggleRecord();
    expect(useTabStore.getState().getActiveTab().recordOutputs).toBe(true);
  });

  it('setLastRunId stores per-tab run id', () => {
    const tabId = useTabStore.getState().activeTabId!;
    useTabStore.getState().setLastRunId(tabId, 'abc-123');
    expect(useTabStore.getState().getActiveTab().lastRunId).toBe('abc-123');
  });

  it('setActiveSegment and addSegmentGroup work independently', () => {
    const seg = { id: 'g1', headNodeId: 'n1', tailNodeId: 'n2' };
    useTabStore.getState().addSegmentGroup(seg);
    useTabStore.getState().setActiveSegment(seg);
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.segmentGroups).toHaveLength(1);
    expect(tab.segmentGroups[0]).toEqual(seg);
    expect(tab.activeSegment).toEqual(seg);
  });

  it('removeSegmentGroup clears active segment when ids match', () => {
    const seg = { id: 'g1', headNodeId: 'n1', tailNodeId: 'n2' };
    useTabStore.getState().addSegmentGroup(seg);
    useTabStore.getState().setActiveSegment(seg);
    useTabStore.getState().removeSegmentGroup('g1');
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.segmentGroups).toHaveLength(0);
    expect(tab.activeSegment).toBeNull();
  });

  it('addSegmentGroup replaces existing group with the same id', () => {
    useTabStore.getState().addSegmentGroup({ id: 'g1', headNodeId: 'a', tailNodeId: 'b' });
    useTabStore.getState().addSegmentGroup({ id: 'g1', headNodeId: 'c', tailNodeId: 'd' });
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.segmentGroups).toHaveLength(1);
    expect(tab.segmentGroups[0].headNodeId).toBe('c');
  });

  it('getSerializedGraph includes segmentGroups', () => {
    useTabStore.getState().addSegmentGroup({ id: 'g1', headNodeId: 'a', tailNodeId: 'b' });
    const serialized = useTabStore.getState().getSerializedGraph();
    expect(serialized.segmentGroups).toHaveLength(1);
    expect(serialized.segmentGroups![0]).toEqual({ id: 'g1', headNodeId: 'a', tailNodeId: 'b' });
  });
});
