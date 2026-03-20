import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { NodeData } from '../../types';
import { getPortColor } from '../../utils';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';

function PresetNode({ id, data, selected }: NodeProps<NodeData>) {
  const openPresetModal = useTabStore((s) => s.openPresetModal);
  const def = data.definition;
  const preset = data.presetDefinition;
  const { t } = useI18n();

  const statusBorderColor =
    data.executionStatus === 'running'
      ? '#FFC107'
      : data.executionStatus === 'completed'
        ? '#4CAF50'
        : data.executionStatus === 'error'
          ? '#F44336'
          : 'transparent';

  const borderColor = selected
    ? '#ffffff'
    : statusBorderColor !== 'transparent'
      ? statusBorderColor
      : '#6B5B00';

  const handleClick = (e: React.MouseEvent) => {
    if (e.detail === 2) {
      openPresetModal(id);
    }
  };

  return (
    <div
      onClick={handleClick}
      style={{
        background: '#1e1e1e',
        border: `2px solid ${borderColor}`,
        borderRadius: 8,
        minWidth: 200,
        fontSize: '0.8125rem',
        color: '#eeeeee',
        boxShadow: selected
          ? '0 0 16px rgba(212,160,23,0.3)'
          : '0 4px 12px rgba(0,0,0,0.4)',
        transition: 'border-color 0.2s, box-shadow 0.2s',
      }}
    >
      {/* Header with gold gradient */}
      <div
        style={{
          background: 'linear-gradient(135deg, #B8860B, #D4A017, #C5941A)',
          padding: '7px 12px',
          borderRadius: '6px 6px 0 0',
          fontWeight: 600,
          fontSize: '0.875rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#1a1a1a' }}>
          {data.label}
        </span>
        <span
          style={{
            fontSize: '0.625rem',
            background: 'rgba(0,0,0,0.3)',
            padding: '2px 5px',
            borderRadius: 3,
            whiteSpace: 'nowrap',
            flexShrink: 0,
            color: '#fff',
            fontWeight: 700,
            letterSpacing: '0.05em',
          }}
        >
          {t('preset.badge')}
        </span>
      </div>

      {/* Ports area */}
      <div style={{ paddingTop: 6, paddingBottom: 6 }}>
        {/* Input handles */}
        {def?.inputs.map((input) => (
          <div
            key={`in-${input.name}`}
            style={{
              position: 'relative',
              padding: '4px 12px 4px 18px',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
            title={input.description}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={input.name}
              style={{
                background: getPortColor(input.data_type),
                width: 10,
                height: 10,
                border: '2px solid #1e1e1e',
                left: -5,
                top: '50%',
                transform: 'translateY(-50%)',
                borderRadius: '50%',
                cursor: 'crosshair',
              }}
            />
            <span
              style={{
                color: getPortColor(input.data_type),
                fontSize: '0.75rem',
                lineHeight: 1,
              }}
            >
              {input.name}
            </span>
          </div>
        ))}

        {/* Divider */}
        {def && def.inputs.length > 0 && def.outputs.length > 0 && (
          <div style={{ height: 1, background: '#333', margin: '4px 0' }} />
        )}

        {/* Output handles */}
        {def?.outputs.map((output) => (
          <div
            key={`out-${output.name}`}
            style={{
              position: 'relative',
              padding: '4px 18px 4px 12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              gap: 6,
            }}
            title={output.description}
          >
            <span
              style={{
                color: getPortColor(output.data_type),
                fontSize: '0.75rem',
                lineHeight: 1,
              }}
            >
              {output.name}
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id={output.name}
              style={{
                background: getPortColor(output.data_type),
                width: 10,
                height: 10,
                border: '2px solid #1e1e1e',
                right: -5,
                top: '50%',
                transform: 'translateY(-50%)',
                borderRadius: '50%',
                cursor: 'crosshair',
              }}
            />
          </div>
        ))}
      </div>

      {/* Footer: node count hint */}
      <div
        style={{
          borderTop: '1px solid #333',
          padding: '5px 10px',
          fontSize: '0.6875rem',
          color: '#888',
          textAlign: 'center',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 4,
        }}
      >
        <span style={{ color: '#D4A017' }}>{preset?.nodes.length ?? 0}</span>
        <span>{t('preset.nodesInside')}</span>
      </div>

      {/* Status footer */}
      {data.executionStatus === 'error' && data.error && (
        <div
          style={{
            padding: '4px 10px',
            fontSize: '0.6875rem',
            color: '#F44336',
            borderTop: '1px solid #333',
            maxWidth: 220,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          title={data.error}
        >
          {t('node.error', { error: data.error })}
        </div>
      )}

      {data.executionStatus === 'running' && (
        <div
          style={{
            padding: '4px 10px',
            fontSize: '0.6875rem',
            color: '#FFC107',
            borderTop: '1px solid #333',
            textAlign: 'center',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 6,
          }}
        >
          <span
            style={{
              display: 'inline-block',
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: '#FFC107',
              animation: 'pulse 1s infinite',
            }}
          />
          {t('node.running')}
        </div>
      )}

      {data.executionStatus === 'completed' && (
        <div
          style={{
            padding: '4px 10px',
            fontSize: '0.6875rem',
            color: '#4CAF50',
            borderTop: '1px solid #333',
            textAlign: 'center',
          }}
        >
          {t('node.completed')}
        </div>
      )}
    </div>
  );
}

export default memo(PresetNode);
