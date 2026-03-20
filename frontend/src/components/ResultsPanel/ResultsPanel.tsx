import { useEffect, useRef } from 'react';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';

function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${String(d.getMilliseconds()).padStart(3, '0')}`;
}

const LOG_TYPE_COLORS = {
  info: '#2196F3',
  error: '#F44336',
  success: '#4CAF50',
} as const;

const LOG_TYPE_BG = {
  info: 'rgba(33,150,243,0.05)',
  error: 'rgba(244,67,54,0.08)',
  success: 'rgba(76,175,80,0.05)',
} as const;

export function ResultsPanel() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const logs = activeTab.logs;
  const clearLogs = useTabStore((s) => s.clearLogs);
  const bottomRef = useRef<HTMLDivElement>(null);
  const { t } = useI18n();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div
      style={{
        height: 200,
        background: '#111',
        borderTop: '1px solid #2a2a2a',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
      }}
    >
      {/* Panel header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '5px 12px',
          borderBottom: '1px solid #1e1e1e',
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              fontSize: '0.6875rem',
              fontWeight: 700,
              color: '#666',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
            }}
          >
            {t('results.title')}
          </span>
          {logs.length > 0 && (
            <span
              style={{
                fontSize: '0.625rem',
                background: '#222',
                border: '1px solid #333',
                borderRadius: 10,
                padding: '1px 6px',
                color: '#666',
              }}
            >
              {logs.length}
            </span>
          )}
        </div>
        <button
          onClick={clearLogs}
          disabled={logs.length === 0}
          style={{
            fontSize: '0.6875rem',
            padding: '2px 8px',
            background: 'transparent',
            border: '1px solid #333',
            borderRadius: 3,
            color: logs.length === 0 ? '#333' : '#666',
            cursor: logs.length === 0 ? 'not-allowed' : 'pointer',
          }}
        >
          {t('results.clear')}
        </button>
      </div>

      {/* Log entries */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '4px 0',
          fontFamily: 'monospace',
        }}
      >
        {logs.length === 0 ? (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: '#333',
              fontSize: '0.75rem',
            }}
          >
            {t('results.empty')}
          </div>
        ) : (
          logs.map((entry, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                alignItems: 'baseline',
                gap: 8,
                padding: '2px 12px',
                background: LOG_TYPE_BG[entry.type],
                borderLeft: `2px solid ${LOG_TYPE_COLORS[entry.type]}`,
                marginBottom: 1,
              }}
            >
              <span
                style={{
                  fontSize: '0.6875rem',
                  color: '#444',
                  flexShrink: 0,
                  lineHeight: 1.6,
                }}
              >
                {formatTimestamp(entry.timestamp)}
              </span>
              {entry.nodeId && (
                <span
                  style={{
                    fontSize: '0.625rem',
                    color: '#555',
                    background: '#1a1a1a',
                    border: '1px solid #2a2a2a',
                    borderRadius: 2,
                    padding: '0 4px',
                    flexShrink: 0,
                    lineHeight: 1.8,
                  }}
                >
                  {String(entry.nodeId).slice(0, 8)}
                </span>
              )}
              {entry.message.startsWith('__IMAGE__:') ? (
                <img
                  src={`data:image/png;base64,${entry.message.slice('__IMAGE__:'.length)}`}
                  alt="output"
                  style={{
                    maxWidth: '100%',
                    maxHeight: 160,
                    borderRadius: 4,
                    border: '1px solid #333',
                    marginTop: 2,
                    marginBottom: 2,
                  }}
                />
              ) : (
                <span
                  style={{
                    fontSize: '0.75rem',
                    color: LOG_TYPE_COLORS[entry.type],
                    lineHeight: 1.6,
                    wordBreak: 'break-word',
                  }}
                >
                  {entry.message}
                </span>
              )}
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
