export interface PortDefinition {
  name: string;
  data_type: string;
  description: string;
  optional: boolean;
}

export interface ParamDefinition {
  name: string;
  param_type: 'int' | 'float' | 'string' | 'bool' | 'select' | 'model_file' | 'image_file';
  default: any;
  description: string;
  options: string[];
  min_value: number | null;
  max_value: number | null;
}

export interface NodeDefinition {
  node_name: string;
  category: string;
  description: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  params: ParamDefinition[];
}

export interface PresetDefinition {
  preset_name: string;
  category: string;
  description: string;
  tags: string[];
  nodes: { id: string; type: string; params: Record<string, any> }[];
  edges: { source: string; sourceHandle: string; target: string; targetHandle: string }[];
  exposed_inputs: { name: string; internal_node: string; internal_port: string; data_type: string; description: string }[];
  exposed_outputs: { name: string; internal_node: string; internal_port: string; data_type: string; description: string }[];
  exposed_params: { internal_node: string; param_name: string; display_name: string; group: string; param_def: ParamDefinition | null }[];
}

export interface NodeProgress {
  event: string;
  epoch?: number;
  total_epochs?: number;
  loss?: number;
  losses?: number[];
  [key: string]: unknown;
}

export interface OutputSummary {
  type: string;
  shape?: number[];
  dtype?: string;
  min?: number;
  max?: number;
  mean?: number;
  value?: any;
  class?: string;
  params?: number;
  trainable?: number;
  repr?: string;
  // Set by the backend when the string value is a file under MODELS_DIR;
  // holds the path relative to MODELS_DIR so the frontend can download it.
  download_path?: string;
}

export interface NodeData {
  label: string;
  type: string;
  params: Record<string, any>;
  definition?: NodeDefinition;
  executionStatus?: ExecutionStatus;
  error?: string;
  progress?: NodeProgress;
  isPreset?: boolean;
  presetDefinition?: PresetDefinition;
  internalParams?: Record<string, Record<string, any>>;
  // Note-specific fields (present only when node.type === 'noteNode')
  noteKind?: 'text' | 'image';
  noteContent?: string;
  noteColor?: string;
  boundToNodeId?: string | null;
  boundOffset?: { x: number; y: number } | null;
  noteWidth?: number;
  noteHeight?: number;
  [key: string]: unknown;
}

export type ExecutionStatus = 'idle' | 'running' | 'completed' | 'error' | 'skipped' | 'cached';

export interface GraphSaveData {
  nodes: any[];
  edges: any[];
  name: string;
  description: string;
  presets?: PresetDefinition[];
}
