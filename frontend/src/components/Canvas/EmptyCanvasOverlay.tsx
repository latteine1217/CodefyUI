import { useNodeDefStore } from '../../store/nodeDefStore';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { DIFFICULTY_COLORS } from '../../styles/theme';
import styles from './EmptyCanvasOverlay.module.css';

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
    <div className={styles.overlay}>
      <div className={styles.inner}>
        <div className={styles.title}>{t('empty.title')}</div>
        <div className={styles.subtitle}>{t('empty.subtitle')}</div>

        {quickStart.length > 0 && (
          <div className={styles.quickStartGrid}>
            {quickStart.map((preset) => {
              const difficulty = preset.tags.find((tag) => tag in DIFFICULTY_COLORS) ?? 'beginner';
              const diffColor = DIFFICULTY_COLORS[difficulty];
              return (
                <button
                  key={preset.preset_name}
                  onClick={() => handleClick(preset)}
                  className={styles.presetCard}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = '#D4A017';
                    e.currentTarget.style.boxShadow = '0 4px 16px rgba(212,160,23,0.15)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = '#3a3a3a';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  <div className={styles.presetCardHeader}>
                    <span className={styles.presetCardName}>{preset.preset_name}</span>
                  </div>
                  <div className={styles.presetCardDesc}>
                    {preset.description.length > 60
                      ? preset.description.slice(0, 60) + '...'
                      : preset.description}
                  </div>
                  <div className={styles.presetCardFooter}>
                    <span
                      className={styles.difficultyBadge}
                      style={{
                        background: `${diffColor}22`,
                        color: diffColor,
                      }}
                    >
                      {difficulty}
                    </span>
                    <span className={styles.nodeCount}>{preset.nodes.length} nodes</span>
                  </div>
                </button>
              );
            })}
          </div>
        )}

        <div className={styles.hint}>{t('empty.hint')}</div>
      </div>
    </div>
  );
}
