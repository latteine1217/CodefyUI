import { useI18n } from '../../i18n';

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
        style={{ position: 'fixed', inset: 0, zIndex: 999 }}
        onClick={onClose}
        onContextMenu={(e) => { e.preventDefault(); onClose(); }}
      />
      {/* Menu */}
      <div
        style={{
          position: 'fixed',
          left: position.x,
          top: position.y,
          zIndex: 1000,
          background: '#222',
          border: '1px solid #444',
          borderRadius: 6,
          minWidth: 160,
          boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
          overflow: 'hidden',
          padding: '4px 0',
        }}
      >
        {items.map((item, i) => (
          <div key={i}>
            <button
              onClick={() => { item.action(); onClose(); }}
              style={{
                display: 'block',
                width: '100%',
                padding: '7px 14px',
                textAlign: 'left',
                background: 'transparent',
                border: 'none',
                color: item.color ?? '#ccc',
                fontSize: '0.8125rem',
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = '#2a2a2a')}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
            >
              {item.label}
            </button>
            {item.dividerAfter && (
              <div style={{ height: 1, background: '#333', margin: '4px 0' }} />
            )}
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
