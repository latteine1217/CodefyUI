import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';
import { computeSegmentNodes } from '../../utils/segmentPath';
import { generateId } from '../../utils';
import { SURFACE, TEXT } from '../../styles/theme';
import styles from './Toolbar.module.css';

const ACCENT = '#ff9500';

export function CompareSegmentButton() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const setActiveSegment = useTabStore((s) => s.setActiveSegment);
  const addToast = useToastStore((s) => s.addToast);
  const { t } = useI18n();

  const selected = activeTab.nodes.filter((n) => n.selected);
  const hasActive = activeTab.activeSegment !== null;

  const handleClick = () => {
    if (hasActive) {
      setActiveSegment(null);
      return;
    }
    if (selected.length !== 2) {
      addToast(t('toolbar.compareSegment.needTwo'), 'warning');
      return;
    }
    const [left, right] =
      selected[0].position.x <= selected[1].position.x
        ? [selected[0], selected[1]]
        : [selected[1], selected[0]];
    const seg = computeSegmentNodes(left.id, right.id, activeTab.nodes, activeTab.edges);
    if (seg.size === 0) {
      addToast(t('segment.noPath'), 'error');
      return;
    }
    setActiveSegment({ id: generateId(), headNodeId: left.id, tailNodeId: right.id });
  };

  const disabled = !hasActive && selected.length !== 2;
  const color = hasActive ? ACCENT : selected.length === 2 ? ACCENT : TEXT.muted;

  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      className={styles.tooltipToggle}
      title={t('toolbar.compareSegment.title')}
      style={{
        color,
        borderColor: hasActive || selected.length === 2 ? ACCENT : SURFACE.borderMedium,
        background: hasActive ? 'rgba(255, 149, 0, 0.1)' : 'transparent',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {t(hasActive ? 'toolbar.clearSegment' : 'toolbar.compareSegment')}
    </button>
  );
}
