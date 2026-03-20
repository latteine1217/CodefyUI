import { useState, useCallback } from 'react';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';

export function TabBar() {
  const tabs = useTabStore((s) => s.tabs);
  const activeTabId = useTabStore((s) => s.activeTabId);
  const addTab = useTabStore((s) => s.addTab);
  const removeTab = useTabStore((s) => s.removeTab);
  const setActiveTab = useTabStore((s) => s.setActiveTab);
  const renameTab = useTabStore((s) => s.renameTab);
  const { t } = useI18n();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');

  const startRename = useCallback((id: string, currentName: string) => {
    setEditingId(id);
    setEditingName(currentName);
  }, []);

  const commitRename = useCallback(() => {
    if (editingId && editingName.trim()) {
      renameTab(editingId, editingName.trim());
    }
    setEditingId(null);
  }, [editingId, editingName, renameTab]);

  const handleClose = useCallback(
    (e: React.MouseEvent, id: string) => {
      e.stopPropagation();
      if (tabs.length <= 1) return;
      const tab = tabs.find((t) => t.id === id);
      if (tab && tab.status === 'running') {
        if (!window.confirm(t('tabs.closeRunning'))) return;
      }
      removeTab(id);
    },
    [tabs, removeTab, t]
  );

  return (
    <div
      style={{
        height: 32,
        background: '#151515',
        borderBottom: '1px solid #2a2a2a',
        display: 'flex',
        alignItems: 'stretch',
        flexShrink: 0,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'stretch',
          flex: 1,
          overflow: 'auto hidden',
        }}
      >
        {tabs.map((tab) => {
          const isActive = tab.id === activeTabId;
          const isRunning = tab.status === 'running';
          const isEditing = editingId === tab.id;

          return (
            <div
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              onDoubleClick={() => startRename(tab.id, tab.name)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '0 12px',
                cursor: 'pointer',
                background: isActive ? '#1e1e1e' : 'transparent',
                borderRight: '1px solid #222',
                borderBottom: isActive ? '2px solid #2196F3' : '2px solid transparent',
                color: isActive ? '#eee' : '#777',
                fontSize: '0.8125rem',
                fontWeight: isActive ? 600 : 400,
                minWidth: 80,
                maxWidth: 180,
                flexShrink: 0,
                transition: 'background 0.15s, color 0.15s',
                userSelect: 'none',
              }}
            >
              {/* Running indicator */}
              {isRunning && (
                <span
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: '50%',
                    background: '#FFC107',
                    boxShadow: '0 0 4px #FFC107',
                    flexShrink: 0,
                  }}
                />
              )}

              {/* Tab name */}
              {isEditing ? (
                <input
                  autoFocus
                  value={editingName}
                  onChange={(e) => setEditingName(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commitRename();
                    if (e.key === 'Escape') setEditingId(null);
                  }}
                  onClick={(e) => e.stopPropagation()}
                  style={{
                    background: '#111',
                    border: '1px solid #444',
                    borderRadius: 3,
                    color: '#eee',
                    fontSize: '0.8125rem',
                    padding: '1px 4px',
                    width: 80,
                    outline: 'none',
                  }}
                />
              ) : (
                <span
                  style={{
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {tab.name}
                </span>
              )}

              {/* Close button */}
              {tabs.length > 1 && (
                <span
                  onClick={(e) => handleClose(e, tab.id)}
                  style={{
                    fontSize: '0.9375rem',
                    color: '#555',
                    cursor: 'pointer',
                    flexShrink: 0,
                    lineHeight: 1,
                    width: 16,
                    height: 16,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRadius: 3,
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = '#ccc')}
                  onMouseLeave={(e) => (e.currentTarget.style.color = '#555')}
                >
                  ×
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Add tab button */}
      <button
        onClick={() => addTab()}
        title={t('tabs.add')}
        style={{
          padding: '0 12px',
          background: 'transparent',
          border: 'none',
          borderLeft: '1px solid #222',
          color: '#555',
          fontSize: '1.0625rem',
          cursor: 'pointer',
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.color = '#ccc')}
        onMouseLeave={(e) => (e.currentTarget.style.color = '#555')}
      >
        +
      </button>
    </div>
  );
}
