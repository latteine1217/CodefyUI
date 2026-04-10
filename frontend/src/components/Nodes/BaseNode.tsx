import { memo, useState } from 'react';
import { Handle, Position, useReactFlow } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { NodeData } from '../../types';
import { getPortColor, isValidConnection } from '../../utils';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { CATEGORY_COLORS, STATUS_COLORS } from '../../styles/theme';
import { ContextMenu, type ContextMenuItem } from '../shared/ContextMenu';
import styles from './BaseNode.module.css';

function BaseNode({ id, data, selected }: NodeProps<NodeData>) {
  const openSubgraphModal = useTabStore((s) => s.openSubgraphModal);
  const toggleEntryPoint = useTabStore((s) => s.toggleEntryPoint);
  const tooltipsEnabled = useUIStore((s) => s.tooltipsEnabled);
  const draggingSourceType = useUIStore((s) => s.draggingSourceType);
  const [hovered, setHovered] = useState(false);
  const [menuPos, setMenuPos] = useState<{ x: number; y: number } | null>(null);
  const { getEdges } = useReactFlow();
  const def = data.definition;
  const category = def?.category ?? 'Utility';
  const headerColor = CATEGORY_COLORS[category] ?? '#607D8B';
  const { t, tn } = useI18n();

  const isSequentialModel = data.type === 'SequentialModel';

  const handleClick = (e: React.MouseEvent) => {
    if (e.detail === 2 && isSequentialModel) {
      openSubgraphModal(id);
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Only show the menu when this node is a DATA root (no incoming
    // data edges). Trigger edges don't count.
    const edges = getEdges();
    const hasDataIncoming = edges.some(
      (edge) => edge.target === id && ((edge.data as { type?: string } | undefined)?.type ?? 'data') === 'data',
    );
    if (hasDataIncoming) return;
    setMenuPos({ x: e.clientX, y: e.clientY });
  };

  const menuItems: ContextMenuItem[] = [
    {
      id: 'toggle',
      label: data.isEntryPoint ? 'Remove Entry Point' : 'Set as Entry Point',
    },
  ];

  // Dynamic border: selected > execution status > default
  const statusBorderColor =
    data.executionStatus === 'running'
      ? STATUS_COLORS.running
      : data.executionStatus === 'completed'
        ? STATUS_COLORS.completed
        : data.executionStatus === 'error'
          ? STATUS_COLORS.error
          : data.executionStatus === 'cached'
            ? STATUS_COLORS.cached
            : 'transparent';

  const borderColor = selected
    ? '#ffffff'
    : statusBorderColor !== 'transparent'
      ? statusBorderColor
      : '#444444';

  const description = def ? tn(def.node_name, 'description', def.description) : '';

  return (
    <div
      onClick={handleClick}
      onContextMenu={handleContextMenu}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`${styles.node} ${data.isEntryPoint ? styles.entryPoint : ''}`}
      style={{
        '--border-color': borderColor,
        boxShadow: selected
          ? '0 0 16px rgba(255,255,255,0.15)'
          : '0 4px 12px rgba(0,0,0,0.4)',
        cursor: isSequentialModel ? 'pointer' : undefined,
      } as React.CSSProperties}
    >
      {/* Tooltip */}
      {tooltipsEnabled && hovered && description && (
        <div className={styles.tooltip}>
          <div className={styles.tooltipTitle}>{data.label}</div>
          <div className={styles.tooltipDesc}>{description}</div>
        </div>
      )}

      {/* Header */}
      <div
        className={styles.header}
        style={{ background: headerColor }}
      >
        <span className={styles.headerLabel}>
          {data.label}
        </span>
        <span className={styles.headerCategory}>
          {category}
        </span>
      </div>

      {/* Ports area */}
      <div className={styles.portsArea}>
        {/* Input handles */}
        {def?.inputs.map((input) => (
          <div
            key={`in-${input.name}`}
            className={styles.portRowInput}
            title={input.description}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={input.name}
              className={`${styles.portHandle} ${styles.portHandleInput}${
                draggingSourceType
                  ? isValidConnection(draggingSourceType, input.data_type)
                    ? ` ${styles.portCompatible}`
                    : ` ${styles.portIncompatible}`
                  : ''
              }`}
              style={{ background: getPortColor(input.data_type) }}
            />
            <span
              className={styles.portLabel}
              style={{ color: getPortColor(input.data_type) }}
            >
              {input.name}
            </span>
            {input.optional && (
              <span className={styles.portOptional}>{t('node.opt')}</span>
            )}
          </div>
        ))}

        {/* Divider between inputs and outputs */}
        {def && def.inputs.length > 0 && def.outputs.length > 0 && (
          <div className={styles.divider} />
        )}

        {/* Output handles */}
        {def?.outputs.map((output) => (
          <div
            key={`out-${output.name}`}
            className={styles.portRowOutput}
            title={output.description}
          >
            <span
              className={styles.portLabel}
              style={{ color: getPortColor(output.data_type) }}
            >
              {output.name}
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id={output.name}
              className={`${styles.portHandle} ${styles.portHandleOutput}`}
              style={{ background: getPortColor(output.data_type) }}
            />
          </div>
        ))}
      </div>

      {/* Params display — special view for SequentialModel */}
      {isSequentialModel && (
        <div className={styles.subgraphSection}>
          {(() => {
            let count = 0;
            try { count = JSON.parse(data.params.layers ?? '[]').length; } catch { /* ignore */ }
            return (
              <>
                <div className={styles.subgraphLayerRow}>
                  <span className={styles.subgraphLayerCount}>{count}</span>
                  <span>{t('subgraph.layerCount', { count: '' }).replace(/\s*$/, '')}</span>
                </div>
                <div className={styles.subgraphHint}>
                  {t('subgraph.hint')}
                </div>
              </>
            );
          })()}
        </div>
      )}

      {/* Params display — normal nodes */}
      {!isSequentialModel && def && def.params.length > 0 && (
        <div className={styles.paramsSection}>
          {def.params.map((p) => {
            const val = data.params[p.name] ?? p.default;
            return (
              <div key={p.name} className={styles.paramRow}>
                <span className={styles.paramName}>{p.name}</span>
                <span
                  className={styles.paramValue}
                  title={String(val)}
                >
                  {String(val)}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Status footer — error */}
      {data.executionStatus === 'error' && data.error && (
        <div
          className={`${styles.statusFooter} ${styles.statusError}`}
          title={data.error}
        >
          {t('node.error', { error: data.error })}
        </div>
      )}

      {/* Status footer — running (with optional progress) */}
      {data.executionStatus === 'running' && (
        <div className={`${styles.statusFooter} ${styles.statusRunning}`}>
          {data.progress?.event === 'epoch' ? (
            <div className={styles.progressContainer}>
              <div className={styles.progressInfo}>
                <span>Epoch {data.progress.epoch}/{data.progress.total_epochs}</span>
                <span>Loss: {Number(data.progress.loss).toFixed(4)}</span>
              </div>
              <div className={styles.progressBarTrack}>
                <div
                  className={styles.progressBarFill}
                  style={{ width: `${((data.progress.epoch ?? 0) / (data.progress.total_epochs ?? 1)) * 100}%` }}
                />
              </div>
            </div>
          ) : (
            <>
              <span className={styles.statusRunningDot} />
              {t('node.running')}
            </>
          )}
        </div>
      )}

      {/* Status footer — completed */}
      {data.executionStatus === 'completed' && (
        <div className={`${styles.statusFooter} ${styles.statusCompleted}`}>
          {t('node.completed')}
        </div>
      )}

      {/* Status footer — cached */}
      {data.executionStatus === 'cached' && (
        <div className={`${styles.statusFooter} ${styles.statusCached}`}>
          <span className={styles.statusCachedDot} />
          {t('node.cached')}
        </div>
      )}

      {menuPos && (
        <ContextMenu
          x={menuPos.x}
          y={menuPos.y}
          items={menuItems}
          onSelect={(itemId) => {
            if (itemId === 'toggle') toggleEntryPoint(id);
          }}
          onClose={() => setMenuPos(null)}
        />
      )}
    </div>
  );
}

export default memo(BaseNode);
