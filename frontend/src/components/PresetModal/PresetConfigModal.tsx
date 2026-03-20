import { useState, useEffect, useMemo } from 'react';
import { useTabStore } from '../../store/tabStore';
import { ParamField } from '../shared/ParamField';
import { useI18n } from '../../i18n';

export function PresetConfigModal() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const closePresetModal = useTabStore((s) => s.closePresetModal);
  const updatePresetInternalParam = useTabStore((s) => s.updatePresetInternalParam);
  const { t } = useI18n();

  const presetModalNodeId = activeTab.presetModalNodeId;
  const node = activeTab.nodes.find((n) => n.id === presetModalNodeId);
  const preset = node?.data.presetDefinition;
  const currentInternalParams = node?.data.internalParams ?? {};

  // Local state for editing
  const [localParams, setLocalParams] = useState<Record<string, Record<string, any>>>({});

  useEffect(() => {
    if (currentInternalParams) {
      setLocalParams(JSON.parse(JSON.stringify(currentInternalParams)));
    }
  }, [presetModalNodeId]);

  // Group exposed params
  const groupedParams = useMemo(() => {
    if (!preset) return {};
    const groups: Record<string, typeof preset.exposed_params> = {};
    for (const ep of preset.exposed_params) {
      const group = ep.group || t('preset.generalGroup');
      if (!groups[group]) groups[group] = [];
      groups[group].push(ep);
    }
    return groups;
  }, [preset, t]);

  if (!presetModalNodeId || !node || !preset) return null;

  const handleParamChange = (internalNodeId: string, paramName: string, value: any) => {
    setLocalParams((prev) => ({
      ...prev,
      [internalNodeId]: {
        ...prev[internalNodeId],
        [paramName]: value,
      },
    }));
  };

  const handleApply = () => {
    for (const [internalNodeId, params] of Object.entries(localParams)) {
      for (const [paramName, value] of Object.entries(params)) {
        updatePresetInternalParam(presetModalNodeId, internalNodeId, paramName, value);
      }
    }
    closePresetModal();
  };

  const handleCancel = () => {
    closePresetModal();
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(0,0,0,0.6)',
        backdropFilter: 'blur(4px)',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) handleCancel();
      }}
    >
      <div
        style={{
          background: '#1a1a1a',
          border: '1px solid #333',
          borderRadius: 12,
          width: 520,
          maxHeight: '80vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '2px solid #D4A017',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            flexShrink: 0,
          }}
        >
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span style={{ fontSize: '1.0625rem', fontWeight: 700, color: '#eee' }}>
                {preset.preset_name}
              </span>
              <span
                style={{
                  fontSize: '0.625rem',
                  background: 'rgba(212,160,23,0.2)',
                  color: '#D4A017',
                  padding: '2px 6px',
                  borderRadius: 3,
                  fontWeight: 700,
                }}
              >
                {t('preset.badge')}
              </span>
            </div>
            <div style={{ fontSize: '0.75rem', color: '#777', lineHeight: 1.4 }}>
              {preset.description}
            </div>
          </div>
          <button
            onClick={handleCancel}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#666',
              fontSize: '1.125rem',
              cursor: 'pointer',
              padding: '0 4px',
              lineHeight: 1,
            }}
          >
            ✕
          </button>
        </div>

        {/* Pipeline preview */}
        <div
          style={{
            padding: '10px 20px',
            borderBottom: '1px solid #2a2a2a',
            overflowX: 'auto',
            flexShrink: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, minWidth: 'max-content' }}>
            {preset.nodes.map((n, i) => (
              <div key={n.id} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span
                  style={{
                    fontSize: '0.6875rem',
                    padding: '3px 8px',
                    background: '#252525',
                    border: '1px solid #3a3a3a',
                    borderRadius: 4,
                    color: '#ccc',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {n.type}
                </span>
                {i < preset.nodes.length - 1 && (
                  <span style={{ color: '#555', fontSize: '0.6875rem' }}>→</span>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Params content */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
          {Object.entries(groupedParams).map(([group, params]) => (
            <div key={group} style={{ marginBottom: 18 }}>
              <div
                style={{
                  fontSize: '0.6875rem',
                  fontWeight: 700,
                  color: '#D4A017',
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  marginBottom: 10,
                  paddingBottom: 4,
                  borderBottom: '1px solid #2a2a2a',
                }}
              >
                {group}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {params.map((ep) => {
                  if (!ep.param_def) return null;
                  const val = localParams[ep.internal_node]?.[ep.param_name] ?? ep.param_def.default;
                  return (
                    <div key={`${ep.internal_node}-${ep.param_name}`}>
                      <ParamField
                        param={ep.param_def}
                        value={val}
                        onChange={(_name, value) => handleParamChange(ep.internal_node, ep.param_name, value)}
                        label={ep.display_name}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Footer buttons */}
        <div
          style={{
            padding: '12px 20px',
            borderTop: '1px solid #2a2a2a',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 8,
            flexShrink: 0,
          }}
        >
          <button
            onClick={handleCancel}
            style={{
              padding: '7px 16px',
              background: '#2a2a2a',
              border: '1px solid #444',
              borderRadius: 6,
              color: '#aaa',
              fontSize: '0.8125rem',
              cursor: 'pointer',
            }}
          >
            {t('preset.cancel')}
          </button>
          <button
            onClick={handleApply}
            style={{
              padding: '7px 16px',
              background: '#D4A017',
              border: 'none',
              borderRadius: 6,
              color: '#1a1a1a',
              fontSize: '0.8125rem',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            {t('preset.apply')}
          </button>
        </div>
      </div>
    </div>
  );
}
