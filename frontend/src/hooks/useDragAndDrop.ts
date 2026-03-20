import { useCallback } from 'react';
import { useReactFlow } from '@xyflow/react';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';

export function useDragAndDrop() {
  const { screenToFlowPosition } = useReactFlow();
  const addNode = useTabStore((s) => s.addNode);
  const addPresetNode = useTabStore((s) => s.addPresetNode);
  const definitions = useNodeDefStore((s) => s.definitions);
  const presets = useNodeDefStore((s) => s.presets);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      // Check for preset drop
      const presetName = event.dataTransfer.getData('application/codefyui-preset');
      if (presetName) {
        const preset = presets.find((p) => p.preset_name === presetName);
        if (preset) addPresetNode(preset, position);
        return;
      }

      // Check for node drop
      const nodeType = event.dataTransfer.getData('application/codefyui-node');
      if (nodeType) {
        const definition = definitions.find((d) => d.node_name === nodeType);
        if (definition) addNode(definition, position);
      }
    },
    [definitions, presets, screenToFlowPosition, addNode, addPresetNode]
  );

  return { onDragOver, onDrop };
}
