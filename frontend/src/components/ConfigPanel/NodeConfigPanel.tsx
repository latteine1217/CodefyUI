import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { ParamField } from '../shared/ParamField';

export function NodeConfigPanel() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const nodes = activeTab.nodes;
  const selectedNodeId = activeTab.selectedNodeId;
  const updateNodeParams = useTabStore((s) => s.updateNodeParams);
  const openPresetModal = useTabStore((s) => s.openPresetModal);
  const { t, tn } = useI18n();

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  const def = selectedNode?.data.definition;
  const nodeName = def?.node_name ?? '';

  const handleChange = (paramName: string, value: any) => {
    if (!selectedNodeId) return;
    updateNodeParams(selectedNodeId, { [paramName]: value });
  };

  if (!selectedNode) {
    return (
      <div
        style={{
          width: 300,
          height: '100%',
          background: '#161616',
          borderLeft: '1px solid #2a2a2a',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        <div style={{ textAlign: 'center', color: '#444', padding: 20 }}>
          <div style={{ fontSize: '1.75rem', marginBottom: 8 }}>○</div>
          <div style={{ fontSize: '0.8125rem' }}>{t('config.selectNode')}</div>
        </div>
      </div>
    );
  }

  const isPreset = selectedNode.data.isPreset;
  const category = def?.category ?? 'Utility';
  const CATEGORY_COLORS: Record<string, string> = {
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
  const accentColor = isPreset ? '#D4A017' : (CATEGORY_COLORS[category] ?? '#607D8B');

  return (
    <div
      style={{
        width: 300,
        height: '100%',
        background: '#161616',
        borderLeft: '1px solid #2a2a2a',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        overflow: 'hidden',
      }}
    >
      {/* Panel header */}
      <div
        style={{
          padding: '12px 14px',
          borderBottom: `2px solid ${accentColor}`,
          flexShrink: 0,
          background: '#1a1a1a',
        }}
      >
        <div
          style={{
            fontSize: '0.75rem',
            fontWeight: 700,
            color: '#888',
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            marginBottom: 4,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          {t('config.title')}
          {isPreset && (
            <span
              style={{
                fontSize: '0.625rem',
                background: 'rgba(212,160,23,0.2)',
                color: '#D4A017',
                padding: '1px 5px',
                borderRadius: 3,
                fontWeight: 600,
              }}
            >
              {t('preset.badge')}
            </span>
          )}
        </div>
        <div
          style={{
            fontSize: '1rem',
            fontWeight: 600,
            color: '#eee',
          }}
        >
          {selectedNode.data.label}
        </div>
        <div
          style={{
            fontSize: '0.75rem',
            color: accentColor,
            marginTop: 2,
          }}
        >
          {category}
        </div>
        {def?.description && (
          <div
            style={{
              fontSize: '0.75rem',
              color: '#666',
              marginTop: 6,
              lineHeight: 1.4,
            }}
          >
            {tn(nodeName, 'description', def.description)}
          </div>
        )}
      </div>

      {/* Scrollable content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 14px' }}>
        {/* Preset: Configure button */}
        {isPreset && (
          <div style={{ marginBottom: 16 }}>
            <div
              style={{
                fontSize: '0.75rem',
                color: '#888',
                marginBottom: 8,
                lineHeight: 1.4,
              }}
            >
              {t('preset.nodeCount', { count: selectedNode.data.presetDefinition?.nodes.length ?? 0 })}
            </div>
            <button
              onClick={() => openPresetModal(selectedNode.id)}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(212,160,23,0.15)',
                border: '1px solid #D4A017',
                borderRadius: 6,
                color: '#D4A017',
                fontSize: '0.8125rem',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              {t('preset.configure')}
            </button>
          </div>
        )}

        {/* Params section (for non-preset nodes) */}
        {!isPreset && def && def.params.length > 0 ? (
          <div>
            <div
              style={{
                fontSize: '0.6875rem',
                fontWeight: 700,
                color: '#666',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                marginBottom: 10,
              }}
            >
              {t('config.parameters')}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {def.params.map((param) => (
                <div key={param.name}>
                  <ParamField
                    param={param}
                    value={selectedNode.data.params[param.name]}
                    onChange={handleChange}
                  />
                  {param.description && (
                    <div
                      style={{
                        fontSize: '0.6875rem',
                        color: '#555',
                        marginTop: 2,
                        lineHeight: 1.3,
                      }}
                    >
                      {tn(nodeName, `param.${param.name}`, param.description)}
                    </div>
                  )}
                  {(param.min_value !== null || param.max_value !== null) && (
                    <div style={{ fontSize: '0.6875rem', color: '#555', marginTop: 2 }}>
                      {t('config.range', {
                        min: param.min_value !== null ? param.min_value : '-∞',
                        max: param.max_value !== null ? param.max_value : '+∞',
                      })}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : !isPreset ? (
          <div
            style={{
              textAlign: 'center',
              color: '#444',
              fontSize: '0.8125rem',
              paddingTop: 20,
            }}
          >
            {t('config.noParams')}
          </div>
        ) : null}

        {/* I/O info section */}
        {def && (def.inputs.length > 0 || def.outputs.length > 0) && (
          <div style={{ marginTop: 20 }}>
            <div
              style={{
                fontSize: '0.6875rem',
                fontWeight: 700,
                color: '#666',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                marginBottom: 8,
              }}
            >
              {t('config.ports')}
            </div>

            {def.inputs.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: '0.6875rem', color: '#555', marginBottom: 4 }}>{t('config.inputs')}</div>
                {def.inputs.map((inp) => (
                  <div
                    key={inp.name}
                    style={{
                      fontSize: '0.75rem',
                      color: '#888',
                      padding: '2px 0',
                      display: 'flex',
                      gap: 6,
                      alignItems: 'center',
                    }}
                  >
                    <span
                      style={{
                        display: 'inline-block',
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        background: '#555',
                        flexShrink: 0,
                      }}
                    />
                    <span style={{ color: '#aaa' }}>{inp.name}</span>
                    <span style={{ color: '#555', fontSize: '0.6875rem' }}>{inp.data_type}</span>
                    {inp.optional && (
                      <span style={{ color: '#444', fontSize: '0.625rem' }}>{t('config.optional')}</span>
                    )}
                  </div>
                ))}
              </div>
            )}

            {def.outputs.length > 0 && (
              <div>
                <div style={{ fontSize: '0.6875rem', color: '#555', marginBottom: 4 }}>{t('config.outputs')}</div>
                {def.outputs.map((out) => (
                  <div
                    key={out.name}
                    style={{
                      fontSize: '0.75rem',
                      color: '#888',
                      padding: '2px 0',
                      display: 'flex',
                      gap: 6,
                      alignItems: 'center',
                    }}
                  >
                    <span
                      style={{
                        display: 'inline-block',
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        background: '#555',
                        flexShrink: 0,
                      }}
                    />
                    <span style={{ color: '#aaa' }}>{out.name}</span>
                    <span style={{ color: '#555', fontSize: '0.6875rem' }}>{out.data_type}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Execution status */}
        {selectedNode.data.executionStatus && selectedNode.data.executionStatus !== 'idle' && (
          <div style={{ marginTop: 20 }}>
            <div
              style={{
                fontSize: '0.6875rem',
                fontWeight: 700,
                color: '#666',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                marginBottom: 6,
              }}
            >
              {t('config.execution')}
            </div>
            <div
              style={{
                padding: '6px 10px',
                borderRadius: 4,
                background:
                  selectedNode.data.executionStatus === 'error'
                    ? 'rgba(244,67,54,0.1)'
                    : selectedNode.data.executionStatus === 'completed'
                      ? 'rgba(76,175,80,0.1)'
                      : 'rgba(255,193,7,0.1)',
                border: `1px solid ${
                  selectedNode.data.executionStatus === 'error'
                    ? '#F44336'
                    : selectedNode.data.executionStatus === 'completed'
                      ? '#4CAF50'
                      : '#FFC107'
                }`,
                fontSize: '0.75rem',
                color:
                  selectedNode.data.executionStatus === 'error'
                    ? '#F44336'
                    : selectedNode.data.executionStatus === 'completed'
                      ? '#4CAF50'
                      : '#FFC107',
              }}
            >
              {selectedNode.data.executionStatus === 'error' && selectedNode.data.error
                ? t('node.error', { error: selectedNode.data.error })
                : t(`status.${selectedNode.data.executionStatus}` as const)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
