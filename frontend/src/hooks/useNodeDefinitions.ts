import { useEffect } from 'react';
import { useNodeDefStore } from '../store/nodeDefStore';

export function useNodeDefinitions() {
  const definitions = useNodeDefStore((s) => s.definitions);
  const categorized = useNodeDefStore((s) => s.categorized);
  const loading = useNodeDefStore((s) => s.loading);
  const error = useNodeDefStore((s) => s.error);
  const fetchDefinitions = useNodeDefStore((s) => s.fetchDefinitions);

  useEffect(() => {
    // Always read the latest store state to avoid closure-captured stale flags.
    const state = useNodeDefStore.getState();
    if (state.definitions.length === 0 && !state.loading) {
      state.fetchDefinitions();
    }
  }, []);

  return { definitions, categorized, loading, error, refetch: fetchDefinitions };
}
