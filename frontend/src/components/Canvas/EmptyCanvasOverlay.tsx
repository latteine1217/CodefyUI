import { useNodeDefStore } from '../../store/nodeDefStore';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';

const DIFFICULTY_COLORS: Record<string, string> = {
  beginner: '#4CAF50',
  intermediate: '#FF9800',
  advanced: '#F44336',
};

export function EmptyCanvasOverlay() {
  const presets = useNodeDefStore((s) => s.presets);
  const addPresetNode = useTabStore((s) => s.addPresetNode);
  const { t } = useI18n();

  // Show up to 3 presets
  const quickStart = presets.slice(0, 3);

  const handleClick = (preset: typeof presets[0]) => {
    addPresetNode(preset, { x: 300, y: 200 });
  };

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 5,
        pointerEvents: 'none',
      }}
    >
      <div style={{ textAlign: 'center', pointerEvents: 'auto' }}>
        <div
          style={{
            fontSize: '1.25rem',
            fontWeight: 700,
            color: '#555',
            marginBottom: 6,
          }}
        >
          {t('empty.title')}
        </div>
        <div
          style={{
            fontSize: '0.875rem',
            color: '#444',
            marginBottom: 24,
          }}
        >
          {t('empty.subtitle')}
        </div>

        {quickStart.length > 0 && (
          <div
            style={{
              display: 'flex',
              gap: 12,
              justifyContent: 'center',
              flexWrap: 'wrap',
              marginBottom: 20,
            }}
          >
            {quickStart.map((preset) => {
              const difficulty = preset.tags.find((tag) => tag in DIFFICULTY_COLORS) ?? 'beginner';
              return (
                <button
                  key={preset.preset_name}
                  onClick={() => handleClick(preset)}
                  style={{
                    background: '#1e1e1e',
                    border: '1px solid #3a3a3a',
                    borderRadius: 8,
                    padding: '14px 18px',
                    cursor: 'pointer',
                    width: 180,
                    textAlign: 'left',
                    transition: 'border-color 0.2s, box-shadow 0.2s',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = '#D4A017';
                    e.currentTarget.style.boxShadow = '0 4px 16px rgba(212,160,23,0.15)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = '#3a3a3a';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#ddd' }}>
                      {preset.preset_name}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.6875rem', color: '#777', lineHeight: 1.4, marginBottom: 8 }}>
                    {preset.description.length > 60
                      ? preset.description.slice(0, 60) + '...'
                      : preset.description}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span
                      style={{
                        fontSize: '0.625rem',
                        padding: '1px 5px',
                        borderRadius: 3,
                        background: `${DIFFICULTY_COLORS[difficulty]}22`,
                        color: DIFFICULTY_COLORS[difficulty],
                        fontWeight: 600,
                      }}
                    >
                      {difficulty}
                    </span>
                    <span style={{ fontSize: '0.625rem', color: '#555' }}>
                      {preset.nodes.length} nodes
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}

        <div style={{ fontSize: '0.75rem', color: '#444' }}>{t('empty.hint')}</div>
      </div>
    </div>
  );
}
