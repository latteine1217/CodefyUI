import { useEffect } from 'react';
import { useNodeDefStore } from '../store/nodeDefStore';

export function useNodeDefinitions() {
  const { definitions, categorized, loading, error, fetchDefinitions } = useNodeDefStore();

  useEffect(() => {
    if (definitions.length === 0 && !loading) {
      fetchDefinitions();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { definitions, categorized, loading, error, refetch: fetchDefinitions };
}
