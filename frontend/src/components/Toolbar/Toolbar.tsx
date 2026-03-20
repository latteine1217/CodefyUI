import { useState, useCallback, useRef } from 'react';
import { useGraphExecution } from '../../hooks/useGraphExecution';
import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { saveGraph, loadGraph, listGraphs, createPreset } from '../../api/rest';
import { useI18n, SUPPORTED_LOCALES } from '../../i18n';
import { resolveSerializedNodes, resolveSerializedEdges } from '../../utils';

function ToolbarButton({
  onClick,
  disabled,
  style,
  children,
  title,
}: {
  onClick: () => void;
  disabled?: boolean;
  style?: React.CSSProperties;
  children: React.ReactNode;
  title?: string;
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        padding: '6px 14px',
        borderRadius: 5,
        border: '1px solid',
        cursor: disabled ? 'not-allowed' : 'pointer',
        fontSize: '0.8125rem',
        fontWeight: 600,
        transition: 'all 0.15s',
        opacity: disabled ? 0.4 : 1,
        transform: hovered && !disabled ? 'translateY(-1px)' : 'none',
        ...style,
      }}
    >
      {children}
    </button>
  );
}

export function Toolbar() {
  const { execute, stop } = useGraphExecution();
  const { clear, getSerializedGraph, setNodes, setEdges } = useTabStore();
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const status = activeTab.status;
  const { reload, fetchDefinitions } = useNodeDefStore();
  const { t, locale, setLocale } = useI18n();

  const [loadMenuOpen, setLoadMenuOpen] = useState(false);
  const [savedGraphs, setSavedGraphs] = useState<{ name: string; file: string }[]>([]);
  const [loadingGraphs, setLoadingGraphs] = useState(false);
  const [langMenuOpen, setLangMenuOpen] = useState(false);

  const isRunning = status === 'running';

  const handleRun = useCallback(() => {
    execute();
  }, [execute]);

  const handleStop = useCallback(() => {
    stop();
  }, [stop]);

  const handleClear = useCallback(() => {
    if (window.confirm(t('toolbar.clear.confirm'))) {
      clear();
    }
  }, [clear, t]);

  const handleSave = useCallback(async () => {
    const name = window.prompt(t('toolbar.save.prompt'));
    if (!name?.trim()) return;
    try {
      const { nodes, edges } = getSerializedGraph();
      await saveGraph({ nodes, edges, name: name.trim(), description: '' });
      window.alert(t('toolbar.save.success', { name: name.trim() }));
    } catch (e) {
      window.alert(t('toolbar.save.fail', { error: (e as Error).message }));
    }
  }, [getSerializedGraph, t]);

  const handleExportSubgraph = useCallback(async () => {
    const { nodes, edges } = getSerializedGraph();
    if (nodes.length === 0) {
      window.alert(t('toolbar.export.empty'));
      return;
    }
    const name = window.prompt(t('toolbar.export.prompt'));
    if (!name?.trim()) return;
    try {
      await createPreset({ name: name.trim(), nodes, edges });
      await fetchDefinitions();
      window.alert(t('toolbar.export.success', { name: name.trim() }));
    } catch (e) {
      window.alert(t('toolbar.export.fail', { error: (e as Error).message }));
    }
  }, [getSerializedGraph, fetchDefinitions, t]);

  const handleLoadOpen = useCallback(async () => {
    setLoadingGraphs(true);
    setLoadMenuOpen(true);
    try {
      const result = await listGraphs();
      setSavedGraphs(Array.isArray(result) ? result : []);
    } catch {
      setSavedGraphs([]);
    } finally {
      setLoadingGraphs(false);
    }
  }, []);

  const handleLoadGraph = useCallback(
    async (name: string) => {
      setLoadMenuOpen(false);
      try {
        const graphData = await loadGraph(name);
        const rawNodes = graphData.nodes ?? [];
        const rawEdges = graphData.edges ?? [];
        const { definitions, presets } = useNodeDefStore.getState();
        setNodes(resolveSerializedNodes(rawNodes, definitions, presets));
        setEdges(resolveSerializedEdges(rawEdges));
      } catch (e) {
        window.alert(t('toolbar.load.fail', { error: (e as Error).message }));
      }
    },
    [setNodes, setEdges, t]
  );

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImportFile = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target?.result as string);
          const rawNodes = data.nodes ?? [];
          const edges = data.edges ?? [];
          if (!Array.isArray(rawNodes) || !Array.isArray(edges)) {
            throw new Error('Invalid graph format');
          }
          const { definitions, presets } = useNodeDefStore.getState();
          setNodes(resolveSerializedNodes(rawNodes, definitions, presets));
          setEdges(resolveSerializedEdges(edges));
        } catch (err) {
          window.alert(t('toolbar.import.fail', { error: (err as Error).message }));
        }
      };
      reader.readAsText(file);
      // Reset so the same file can be re-imported
      event.target.value = '';
    },
    [setNodes, setEdges, t]
  );

  const handleReloadNodes = useCallback(async () => {
    try {
      await reload();
    } catch (e) {
      window.alert(t('toolbar.reload.fail', { error: (e as Error).message }));
    }
  }, [reload, t]);

  const statusKey = `status.${status}` as const;

  return (
    <div
      style={{
        height: 48,
        background: '#1a1a1a',
        borderBottom: '1px solid #2a2a2a',
        display: 'flex',
        alignItems: 'center',
        padding: '0 16px',
        gap: 8,
        flexShrink: 0,
        position: 'relative',
        zIndex: 100,
      }}
    >
      {/* Logo */}
      <div
        style={{
          fontSize: '1rem',
          fontWeight: 800,
          color: '#eee',
          marginRight: 12,
          letterSpacing: '-0.02em',
        }}
      >
        <span style={{ color: '#2196F3' }}>Codefy</span>
        <span style={{ color: '#888' }}>UI</span>
      </div>

      <div style={{ width: 1, height: 24, background: '#333' }} />

      {/* Run */}
      <ToolbarButton
        onClick={handleRun}
        disabled={isRunning}
        title={t('toolbar.run.title')}
        style={{
          background: isRunning ? '#1a3d1a' : '#1b5e20',
          borderColor: '#4CAF50',
          color: '#4CAF50',
        }}
      >
        {isRunning ? t('toolbar.running') : t('toolbar.run')}
      </ToolbarButton>

      {/* Stop */}
      <ToolbarButton
        onClick={handleStop}
        disabled={!isRunning}
        title={t('toolbar.stop.title')}
        style={{
          background: isRunning ? '#3d1a1a' : 'transparent',
          borderColor: '#F44336',
          color: '#F44336',
        }}
      >
        {t('toolbar.stop')}
      </ToolbarButton>

      <div style={{ width: 1, height: 24, background: '#333' }} />

      {/* Save */}
      <ToolbarButton
        onClick={handleSave}
        title={t('toolbar.save.title')}
        style={{
          background: 'transparent',
          borderColor: '#555',
          color: '#ccc',
        }}
      >
        {t('toolbar.save')}
      </ToolbarButton>

      {/* Export as Subgraph */}
      <ToolbarButton
        onClick={handleExportSubgraph}
        title={t('toolbar.export.title')}
        style={{
          background: 'transparent',
          borderColor: '#D4A017',
          color: '#D4A017',
        }}
      >
        {t('toolbar.export')}
      </ToolbarButton>

      {/* Load */}
      <div style={{ position: 'relative' }}>
        <ToolbarButton
          onClick={handleLoadOpen}
          title={t('toolbar.load.title')}
          style={{
            background: 'transparent',
            borderColor: '#555',
            color: '#ccc',
          }}
        >
          {t('toolbar.load')}
        </ToolbarButton>

        {loadMenuOpen && (
          <>
            <div
              style={{ position: 'fixed', inset: 0, zIndex: 199 }}
              onClick={() => setLoadMenuOpen(false)}
            />
            <div
              style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                marginTop: 4,
                background: '#222',
                border: '1px solid #444',
                borderRadius: 6,
                minWidth: 200,
                zIndex: 200,
                boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
                overflow: 'hidden',
              }}
            >
              {loadingGraphs ? (
                <div style={{ padding: '10px 12px', fontSize: '0.8125rem', color: '#666' }}>
                  {t('toolbar.load.loading')}
                </div>
              ) : savedGraphs.length === 0 ? (
                <div style={{ padding: '10px 12px', fontSize: '0.8125rem', color: '#555' }}>
                  {t('toolbar.load.empty')}
                </div>
              ) : (
                savedGraphs.map((g) => (
                  <button
                    key={g.file}
                    onClick={() => handleLoadGraph(g.file)}
                    style={{
                      display: 'block',
                      width: '100%',
                      padding: '8px 12px',
                      textAlign: 'left',
                      background: 'transparent',
                      border: 'none',
                      borderBottom: '1px solid #333',
                      color: '#ccc',
                      fontSize: '0.8125rem',
                      cursor: 'pointer',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = '#2a2a2a')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    {g.name}
                  </button>
                ))
              )}
              {/* Divider + Upload */}
              <div style={{ height: 1, background: '#444', margin: '2px 0' }} />
              <button
                onClick={() => {
                  setLoadMenuOpen(false);
                  fileInputRef.current?.click();
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  width: '100%',
                  padding: '8px 12px',
                  textAlign: 'left',
                  background: 'transparent',
                  border: 'none',
                  color: '#2196F3',
                  fontSize: '0.8125rem',
                  cursor: 'pointer',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = '#2a2a2a')}
                onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
              >
                {t('toolbar.import')}
              </button>
            </div>
          </>
        )}
      </div>

      {/* Hidden file input for JSON import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        style={{ display: 'none' }}
        onChange={handleImportFile}
      />

      {/* Clear */}
      <ToolbarButton
        onClick={handleClear}
        title={t('toolbar.clear.title')}
        style={{
          background: 'transparent',
          borderColor: '#555',
          color: '#999',
        }}
      >
        {t('toolbar.clear')}
      </ToolbarButton>

      <div style={{ width: 1, height: 24, background: '#333' }} />

      {/* Reload nodes */}
      <ToolbarButton
        onClick={handleReloadNodes}
        title={t('toolbar.reloadNodes.title')}
        style={{
          background: 'transparent',
          borderColor: '#444',
          color: '#777',
        }}
      >
        {t('toolbar.reloadNodes')}
      </ToolbarButton>

      {/* Right side: status + language */}
      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
        {/* Status indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              display: 'inline-block',
              width: 8,
              height: 8,
              borderRadius: '50%',
              background:
                status === 'running'
                  ? '#FFC107'
                  : status === 'completed'
                    ? '#4CAF50'
                    : status === 'error'
                      ? '#F44336'
                      : '#444',
              boxShadow: status === 'running' ? '0 0 6px #FFC107' : 'none',
            }}
          />
          <span
            style={{
              fontSize: '0.75rem',
              color:
                status === 'running'
                  ? '#FFC107'
                  : status === 'completed'
                    ? '#4CAF50'
                    : status === 'error'
                      ? '#F44336'
                      : '#555',
            }}
          >
            {t(statusKey)}
          </span>
        </div>

        {/* Language selector */}
        <div style={{ position: 'relative' }}>
          <button
            onClick={() => setLangMenuOpen((v) => !v)}
            style={{
              padding: '3px 8px',
              borderRadius: 4,
              border: '1px solid #444',
              background: langMenuOpen ? '#333' : '#222',
              color: '#aaa',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              minWidth: 32,
              textAlign: 'center',
            }}
          >
            {SUPPORTED_LOCALES.find((l) => l.code === locale)?.label ?? locale}
          </button>

          {langMenuOpen && (
            <>
              <div
                style={{ position: 'fixed', inset: 0, zIndex: 199 }}
                onClick={() => setLangMenuOpen(false)}
              />
              <div
                style={{
                  position: 'absolute',
                  top: '100%',
                  right: 0,
                  marginTop: 6,
                  background: '#222',
                  border: '1px solid #444',
                  borderRadius: 6,
                  minWidth: 140,
                  zIndex: 200,
                  boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
                  overflow: 'hidden',
                  padding: '4px 0',
                }}
              >
                {SUPPORTED_LOCALES.map((l) => (
                  <button
                    key={l.code}
                    onClick={() => { setLocale(l.code); setLangMenuOpen(false); }}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      width: '100%',
                      padding: '7px 12px',
                      background: 'transparent',
                      border: 'none',
                      color: l.code === locale ? '#2196F3' : '#ccc',
                      fontSize: '0.8125rem',
                      cursor: 'pointer',
                      fontWeight: l.code === locale ? 600 : 400,
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = '#2a2a2a')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    <span>{l.nativeName}</span>
                    {l.code === locale && <span style={{ fontSize: '0.6875rem' }}>✓</span>}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
