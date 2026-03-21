import { useI18n } from '../../i18n';
import styles from './NodeContextMenu.module.css';

export interface ContextMenuPosition {
  nodeId: string;
  x: number;
  y: number;
}

interface MenuItem {
  label: string;
  action: () => void;
  color?: string;
  dividerAfter?: boolean;
}

interface NodeContextMenuProps {
  position: ContextMenuPosition;
  items: MenuItem[];
  onClose: () => void;
}

export function NodeContextMenu({ position, items, onClose }: NodeContextMenuProps) {
  return (
    <>
      {/* Backdrop */}
      <div
        className={styles.backdrop}
        onClick={onClose}
        onContextMenu={(e) => { e.preventDefault(); onClose(); }}
      />
      {/* Menu — left/top are dynamic from position */}
      <div
        className={styles.menu}
        style={{ left: position.x, top: position.y }}
      >
        {items.map((item, i) => (
          <div key={i}>
            <button
              onClick={() => { item.action(); onClose(); }}
              className={styles.menuItem}
              style={{ color: item.color ?? '#ccc' }}
            >
              {item.label}
            </button>
            {item.dividerAfter && <div className={styles.divider} />}
          </div>
        ))}
      </div>
    </>
  );
}

export function useNodeContextMenuItems(
  nodeId: string,
  callbacks: {
    onDelete: (id: string) => void;
    onRename: (id: string) => void;
    onDuplicate: (id: string) => void;
  },
) {
  const { t } = useI18n();

  return [
    { label: t('contextMenu.rename'), action: () => callbacks.onRename(nodeId), dividerAfter: false },
    { label: t('contextMenu.duplicate'), action: () => callbacks.onDuplicate(nodeId), dividerAfter: true },
    { label: t('contextMenu.delete'), action: () => callbacks.onDelete(nodeId), color: '#F44336' },
  ];
}
