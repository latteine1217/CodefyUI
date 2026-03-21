import { useEffect, useRef } from 'react';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import styles from './ResultsPanel.module.css';

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
    <div className={styles.panel}>
      {/* Panel header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.title}>{t('results.title')}</span>
          {logs.length > 0 && (
            <span className={styles.countBadge}>{logs.length}</span>
          )}
        </div>
        <button
          onClick={clearLogs}
          disabled={logs.length === 0}
          className={`${styles.clearBtn} ${logs.length === 0 ? styles.clearBtnDisabled : styles.clearBtnEnabled}`}
        >
          {t('results.clear')}
        </button>
      </div>

      {/* Log entries */}
      <div className={styles.logArea}>
        {logs.length === 0 ? (
          <div className={styles.emptyState}>{t('results.empty')}</div>
        ) : (
          logs.map((entry, i) => (
            <div
              key={i}
              className={styles.logEntry}
              style={{
                background: LOG_TYPE_BG[entry.type],
                borderLeft: `2px solid ${LOG_TYPE_COLORS[entry.type]}`,
              }}
            >
              <span className={styles.timestamp}>
                {formatTimestamp(entry.timestamp)}
              </span>
              {entry.nodeId && (
                <span className={styles.nodeIdBadge}>
                  {String(entry.nodeId).slice(0, 8)}
                </span>
              )}
              {entry.message.startsWith('__IMAGE__:') ? (
                <img
                  src={`data:image/png;base64,${entry.message.slice('__IMAGE__:'.length)}`}
                  alt="output"
                  className={styles.logImage}
                />
              ) : (
                <span
                  className={styles.logMessage}
                  style={{ color: LOG_TYPE_COLORS[entry.type] }}
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
