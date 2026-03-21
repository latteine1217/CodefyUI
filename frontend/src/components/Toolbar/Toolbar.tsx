import { useState, useCallback, useRef } from 'react';
import { useGraphExecution } from '../../hooks/useGraphExecution';
import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { saveGraph, loadGraph, listGraphs, createPreset } from '../../api/rest';
import { useI18n, SUPPORTED_LOCALES } from '../../i18n';
import { resolveSerializedNodes, resolveSerializedEdges } from '../../utils';
import { SURFACE, TEXT, BRAND, STATUS_COLORS } from '../../styles/theme';
import styles from './Toolbar.module.css';

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
      className={styles.button}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
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
      const { nodes, edges, presets } = getSerializedGraph();
      await saveGraph({ nodes, edges, name: name.trim(), description: '', presets });
      window.alert(t('toolbar.save.success', { name: name.trim() }));
    } catch (e) {
      window.alert(t('toolbar.save.fail', { error: (e as Error).message }));
    }
  }, [getSerializedGraph, t]);

  const handleExportJson = useCallback(() => {
    const { nodes, edges, presets } = getSerializedGraph();
    if (nodes.length === 0) {
      window.alert(t('toolbar.exportJson.empty'));
      return;
    }
    const name = activeTab.name || 'graph';
    const data = { name, description: '', nodes, edges, presets };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [getSerializedGraph, activeTab.name, t]);

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
        const store = useNodeDefStore.getState();
        // Merge embedded presets from the saved graph
        const savedPresets = Array.isArray(graphData.presets) ? graphData.presets : [];
        const mergedPresets = [...store.presets];
        for (const p of savedPresets) {
          if (!mergedPresets.some((ep) => ep.preset_name === p.preset_name)) {
            mergedPresets.push(p);
          }
        }
        setNodes(resolveSerializedNodes(rawNodes, store.definitions, mergedPresets));
        setEdges(resolveSerializedEdges(rawEdges));
        if (savedPresets.length > 0) {
          useNodeDefStore.setState({ presets: mergedPresets });
        }
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
          const store = useNodeDefStore.getState();
          // Merge embedded presets from the imported file
          const importedPresets = Array.isArray(data.presets) ? data.presets : [];
          const mergedPresets = [...store.presets];
          for (const p of importedPresets) {
            if (!mergedPresets.some((ep) => ep.preset_name === p.preset_name)) {
              mergedPresets.push(p);
            }
          }
          setNodes(resolveSerializedNodes(rawNodes, store.definitions, mergedPresets));
          setEdges(resolveSerializedEdges(edges));
          // Persist merged presets into the store so preset nodes work correctly
          if (importedPresets.length > 0) {
            useNodeDefStore.setState({ presets: mergedPresets });
          }
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

  // Derive status dot color + glow from theme tokens
  const statusDotColor = STATUS_COLORS[status] ?? SURFACE.borderMedium;
  const statusTextColor = STATUS_COLORS[status] ?? TEXT.dim;
  const statusGlow = status === 'running' ? `0 0 6px ${STATUS_COLORS.running}` : 'none';

  return (
    <div className={styles.root}>
      {/* Logo */}
      <div className={styles.logo}>
        <span className={styles.logoBrand}>Codefy</span>
        <span className={styles.logoSuffix}>UI</span>
      </div>

      <div className={styles.divider} />

      {/* Run */}
      <ToolbarButton
        onClick={handleRun}
        disabled={isRunning}
        title={t('toolbar.run.title')}
        style={{
          background: isRunning ? '#1a3d1a' : '#1b5e20',
          borderColor: BRAND.success,
          color: BRAND.success,
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
          borderColor: BRAND.error,
          color: BRAND.error,
        }}
      >
        {t('toolbar.stop')}
      </ToolbarButton>

      <div className={styles.divider} />

      {/* Save */}
      <ToolbarButton
        onClick={handleSave}
        title={t('toolbar.save.title')}
        style={{
          background: 'transparent',
          borderColor: SURFACE.borderHeavy,
          color: TEXT.secondary,
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
          borderColor: BRAND.preset,
          color: BRAND.preset,
        }}
      >
        {t('toolbar.export')}
      </ToolbarButton>

      {/* Export JSON */}
      <ToolbarButton
        onClick={handleExportJson}
        title={t('toolbar.exportJson.title')}
        style={{
          background: 'transparent',
          borderColor: BRAND.primary,
          color: BRAND.primary,
        }}
      >
        {t('toolbar.exportJson')}
      </ToolbarButton>

      {/* Load */}
      <div style={{ position: 'relative' }}>
        <ToolbarButton
          onClick={handleLoadOpen}
          title={t('toolbar.load.title')}
          style={{
            background: 'transparent',
            borderColor: SURFACE.borderHeavy,
            color: TEXT.secondary,
          }}
        >
          {t('toolbar.load')}
        </ToolbarButton>

        {loadMenuOpen && (
          <>
            <div
              className={styles.overlay}
              onClick={() => setLoadMenuOpen(false)}
            />
            <div className={styles.loadDropdown}>
              {loadingGraphs ? (
                <div className={styles.dropdownMessageMuted}>
                  {t('toolbar.load.loading')}
                </div>
              ) : savedGraphs.length === 0 ? (
                <div className={styles.dropdownMessageDim}>
                  {t('toolbar.load.empty')}
                </div>
              ) : (
                savedGraphs.map((g) => (
                  <button
                    key={g.file}
                    onClick={() => handleLoadGraph(g.file)}
                    className={styles.dropdownItem}
                    onMouseEnter={(e) => (e.currentTarget.style.background = SURFACE.hover)}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    {g.name}
                  </button>
                ))
              )}
              {/* Divider + Upload */}
              <div className={styles.dropdownDivider} />
              <button
                onClick={() => {
                  setLoadMenuOpen(false);
                  fileInputRef.current?.click();
                }}
                className={styles.dropdownImport}
                onMouseEnter={(e) => (e.currentTarget.style.background = SURFACE.hover)}
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
        className={styles.fileInput}
        onChange={handleImportFile}
      />

      {/* Clear */}
      <ToolbarButton
        onClick={handleClear}
        title={t('toolbar.clear.title')}
        style={{
          background: 'transparent',
          borderColor: SURFACE.borderHeavy,
          color: TEXT.tertiary,
        }}
      >
        {t('toolbar.clear')}
      </ToolbarButton>

      <div className={styles.divider} />

      {/* Reload nodes */}
      <ToolbarButton
        onClick={handleReloadNodes}
        title={t('toolbar.reloadNodes.title')}
        style={{
          background: 'transparent',
          borderColor: SURFACE.borderMedium,
          color: '#777777',
        }}
      >
        {t('toolbar.reloadNodes')}
      </ToolbarButton>

      {/* Right side: status + language */}
      <div className={styles.rightCluster}>
        {/* Status indicator */}
        <div className={styles.statusGroup}>
          <span
            className={styles.statusDot}
            style={{
              background: statusDotColor,
              boxShadow: statusGlow,
            }}
          />
          <span
            className={styles.statusLabel}
            style={{ color: statusTextColor }}
          >
            {t(statusKey)}
          </span>
        </div>

        {/* Language selector */}
        <div style={{ position: 'relative' }}>
          <button
            onClick={() => setLangMenuOpen((v) => !v)}
            className={styles.langButton}
            style={{ background: langMenuOpen ? SURFACE.borderLight : SURFACE.input }}
          >
            {SUPPORTED_LOCALES.find((l) => l.code === locale)?.label ?? locale}
          </button>

          {langMenuOpen && (
            <>
              <div
                className={styles.overlay}
                onClick={() => setLangMenuOpen(false)}
              />
              <div className={styles.langDropdown}>
                {SUPPORTED_LOCALES.map((l) => (
                  <button
                    key={l.code}
                    onClick={() => { setLocale(l.code); setLangMenuOpen(false); }}
                    className={styles.langOption}
                    style={{
                      color: l.code === locale ? BRAND.primary : TEXT.secondary,
                      fontWeight: l.code === locale ? 600 : 400,
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = SURFACE.hover)}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    <span>{l.nativeName}</span>
                    {l.code === locale && (
                      <span className={styles.langOptionCheck}>✓</span>
                    )}
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
