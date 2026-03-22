/** Shared design tokens — single source of truth for colors used across components. */

export const CATEGORY_COLORS: Record<string, string> = {
  CNN: '#4CAF50',
  RNN: '#2196F3',
  Transformer: '#9C27B0',
  RL: '#FF9800',
  Data: '#00BCD4',
  Training: '#F44336',
  IO: '#795548',
  'Data Flow': '#FF6F00',
  Utility: '#607D8B',
  Normalization: '#26A69A',
  'Tensor Operations': '#5C6BC0',
};

export const DIFFICULTY_COLORS: Record<string, string> = {
  beginner: '#4CAF50',
  intermediate: '#FF9800',
  advanced: '#F44336',
};

export const STATUS_COLORS: Record<string, string> = {
  running: '#FFC107',
  completed: '#4CAF50',
  error: '#F44336',
  cached: '#2196F3',
  skipped: '#9E9E9E',
  idle: '#444',
};

export const SURFACE = {
  bg: '#121212',
  panel: '#161616',
  card: '#1e1e1e',
  toolbar: '#1a1a1a',
  input: '#222',
  hover: '#2a2a2a',
  border: '#2a2a2a',
  borderLight: '#333',
  borderMedium: '#444',
  borderHeavy: '#555',
} as const;

export const TEXT = {
  primary: '#eee',
  secondary: '#ccc',
  tertiary: '#888',
  muted: '#666',
  dim: '#555',
  dimmer: '#444',
} as const;

export const BRAND = {
  primary: '#2196F3',
  preset: '#D4A017',
  success: '#4CAF50',
  error: '#F44336',
  warning: '#FFC107',
} as const;
