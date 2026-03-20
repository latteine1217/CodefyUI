import { useCallback, useMemo, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  type NodeTypes,
  type OnConnect,
  type IsValidConnection,
  type Connection,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import BaseNode from '../Nodes/BaseNode';
import PresetNode from '../Nodes/PresetNode';
import { EmptyCanvasOverlay } from './EmptyCanvasOverlay';
import {
  NodeContextMenu,
  useNodeContextMenuItems,
  type ContextMenuPosition,
} from '../ContextMenu/NodeContextMenu';
import { useTabStore } from '../../store/tabStore';
import { useDragAndDrop } from '../../hooks/useDragAndDrop';
import { isValidConnection } from '../../utils';
import { useI18n } from '../../i18n';

const nodeTypes: NodeTypes = {
  baseNode: BaseNode,
  presetNode: PresetNode,
};

export function FlowCanvas() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const onNodesChange = useTabStore((s) => s.onNodesChange);
  const onEdgesChange = useTabStore((s) => s.onEdgesChange);
  const storeOnConnect = useTabStore((s) => s.onConnect);
  const setSelectedNodeId = useTabStore((s) => s.setSelectedNodeId);
  const deleteNode = useTabStore((s) => s.deleteNode);
  const duplicateNode = useTabStore((s) => s.duplicateNode);
  const renameNode = useTabStore((s) => s.renameNode);
  const { t } = useI18n();

  const [contextMenu, setContextMenu] = useState<ContextMenuPosition | null>(null);

  const { onDragOver, onDrop } = useDragAndDrop();

  const handleConnect: OnConnect = useCallback(
    (connection) => {
      storeOnConnect(connection);
    },
    [storeOnConnect]
  );

  const handleIsValidConnection: IsValidConnection = useCallback(
    (connection: Connection) => {
      const { source, target, sourceHandle, targetHandle } = connection;
      if (!source || !target) return false;
      if (source === target) return false;

      if (sourceHandle && targetHandle) {
        const { tabs, activeTabId } = useTabStore.getState();
        const tab = tabs.find((t) => t.id === activeTabId)!;
        const sourceNode = tab.nodes.find((n) => n.id === source);
        const targetNode = tab.nodes.find((n) => n.id === target);
        if (!sourceNode || !targetNode) return true;

        const sourceDef = sourceNode.data.definition;
        const targetDef = targetNode.data.definition;
        if (!sourceDef || !targetDef) return true;

        const sourceOutput = sourceDef.outputs.find((o) => o.name === sourceHandle);
        const targetInput = targetDef.inputs.find((i) => i.name === targetHandle);
        if (!sourceOutput || !targetInput) return true;

        return isValidConnection(sourceOutput.data_type, targetInput.data_type);
      }

      return true;
    },
    []
  );

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      setSelectedNodeId(node.id);
    },
    [setSelectedNodeId]
  );

  const handlePaneClick = useCallback(() => {
    setSelectedNodeId(null);
    setContextMenu(null);
  }, [setSelectedNodeId]);

  const handleNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: { id: string }) => {
      event.preventDefault();
      setSelectedNodeId(node.id);
      setContextMenu({ nodeId: node.id, x: event.clientX, y: event.clientY });
    },
    [setSelectedNodeId]
  );

  const handleRename = useCallback(
    (nodeId: string) => {
      const node = activeTab.nodes.find((n) => n.id === nodeId);
      const currentLabel = node?.data.label ?? '';
      const newLabel = window.prompt(t('contextMenu.rename.prompt'), currentLabel);
      if (newLabel !== null && newLabel.trim()) {
        renameNode(nodeId, newLabel.trim());
      }
    },
    [activeTab.nodes, renameNode, t]
  );

  const menuItems = useNodeContextMenuItems(contextMenu?.nodeId ?? '', {
    onDelete: deleteNode,
    onRename: handleRename,
    onDuplicate: duplicateNode,
  });

  const proOptions = useMemo(() => ({ hideAttribution: true }), []);
  const isEmpty = activeTab.nodes.length === 0;

  return (
    <div style={{ width: '100%', height: '100%', background: '#0a0a0a', position: 'relative' }}>
      {isEmpty && <EmptyCanvasOverlay />}
      <ReactFlow
        nodes={activeTab.nodes}
        edges={activeTab.edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={handleConnect}
        isValidConnection={handleIsValidConnection}
        onNodeClick={handleNodeClick}
        onNodeContextMenu={handleNodeContextMenu}
        onPaneClick={handlePaneClick}
        onDragOver={onDragOver}
        onDrop={onDrop}
        nodeTypes={nodeTypes}
        fitView
        proOptions={proOptions}
        deleteKeyCode="Delete"
        multiSelectionKeyCode="Shift"
        style={{ background: '#0a0a0a' }}
        defaultEdgeOptions={{
          animated: false,
          style: { stroke: '#555', strokeWidth: 2 },
        }}
        connectionLineStyle={{ stroke: '#888', strokeWidth: 2 }}
        snapToGrid={false}
      >
        <Background
          color="#2a2a2a"
          variant={BackgroundVariant.Dots}
          gap={24}
          size={1.5}
        />
        <Controls />
        <MiniMap
          style={{
            background: '#1e1e1e',
            border: '1px solid #333',
            borderRadius: 6,
          }}
          nodeColor={(node) => {
            const data = node.data as any;
            if (data?.isPreset) return '#D4A017';
            const category = data?.definition?.category ?? 'Utility';
            const colors: Record<string, string> = {
              CNN: '#4CAF50',
              RNN: '#2196F3',
              Transformer: '#9C27B0',
              RL: '#FF9800',
              Data: '#00BCD4',
              Training: '#F44336',
              IO: '#795548',
              Control: '#FF6F00',
              Utility: '#607D8B',
            };
            return colors[category] ?? '#607D8B';
          }}
          maskColor="rgba(0,0,0,0.7)"
        />
      </ReactFlow>

      {contextMenu && (
        <NodeContextMenu
          position={contextMenu}
          items={menuItems}
          onClose={() => setContextMenu(null)}
        />
      )}
    </div>
  );
}
