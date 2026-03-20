// Re-export tabStore as flowStore for backward compatibility.
// All flow state is now per-tab inside tabStore.
export { useTabStore as useFlowStore } from './tabStore';
