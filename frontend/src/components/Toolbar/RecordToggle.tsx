import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { SURFACE, TEXT } from '../../styles/theme';
import styles from './Toolbar.module.css';

export function RecordToggle() {
  const recording = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    return tab?.recordOutputs ?? true;
  });
  const toggle = useTabStore((s) => s.toggleRecord);
  const { t } = useI18n();

  const onColor = '#e63946';

  return (
    <button
      onClick={toggle}
      className={styles.tooltipToggle}
      title={t('toolbar.record.title')}
      style={{
        color: recording ? onColor : TEXT.muted,
        borderColor: recording ? onColor : SURFACE.borderMedium,
        background: recording ? 'rgba(230, 57, 70, 0.1)' : 'transparent',
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
      }}
    >
      <span
        aria-hidden="true"
        style={{
          display: 'inline-block',
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: recording ? onColor : 'transparent',
          border: `1.5px solid ${recording ? onColor : TEXT.muted}`,
        }}
      />
      {t(recording ? 'toolbar.record.on' : 'toolbar.record.off')}
    </button>
  );
}
