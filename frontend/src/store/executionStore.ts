// Re-export tabStore as executionStore for backward compatibility.
// Execution state is now per-tab inside tabStore.
export { useTabStore as useExecutionStore } from './tabStore';
export type { LogEntry } from './tabStore';
