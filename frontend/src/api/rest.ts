import type { NodeDefinition, GraphSaveData, PresetDefinition } from '../types';

const BASE_URL = '/api';

export async function fetchNodeDefinitions(): Promise<NodeDefinition[]> {
  const res = await fetch(`${BASE_URL}/nodes`);
  if (!res.ok) throw new Error(`Failed to fetch node definitions: ${res.statusText}`);
  return res.json();
}

export async function fetchPresetDefinitions(): Promise<PresetDefinition[]> {
  const res = await fetch(`${BASE_URL}/presets`);
  if (!res.ok) throw new Error(`Failed to fetch presets: ${res.statusText}`);
  return res.json();
}

export async function validateGraph(nodes: any[], edges: any[]) {
  const res = await fetch(`${BASE_URL}/graph/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nodes, edges }),
  });
  if (!res.ok) throw new Error(`Validation failed: ${res.statusText}`);
  return res.json();
}

export async function saveGraph(data: GraphSaveData) {
  const res = await fetch(`${BASE_URL}/graph/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Save failed: ${res.statusText}`);
  return res.json();
}

export async function loadGraph(name: string) {
  const res = await fetch(`${BASE_URL}/graph/load/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`Load failed: ${res.statusText}`);
  return res.json();
}

export async function listGraphs() {
  const res = await fetch(`${BASE_URL}/graph/list`);
  if (!res.ok) throw new Error(`List failed: ${res.statusText}`);
  return res.json();
}

export async function exportGraph(nodes: any[], edges: any[]) {
  const res = await fetch(`${BASE_URL}/graph/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nodes, edges }),
  });
  if (!res.ok) throw new Error(`Export failed: ${res.statusText}`);
  return res.json();
}

export async function createPreset(data: {
  name: string;
  description?: string;
  category?: string;
  tags?: string[];
  nodes: any[];
  edges: any[];
}): Promise<PresetDefinition> {
  const res = await fetch(`${BASE_URL}/presets/create`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Export failed: ${res.statusText}`);
  }
  return res.json();
}

export async function reloadNodes() {
  const res = await fetch(`${BASE_URL}/nodes/reload`, { method: 'POST' });
  if (!res.ok) throw new Error(`Reload failed: ${res.statusText}`);
  return res.json();
}
