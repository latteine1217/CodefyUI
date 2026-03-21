import { useState } from 'react';
import { useNodeDefinitions } from '../../hooks/useNodeDefinitions';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useI18n } from '../../i18n';
import type { NodeDefinition, PresetDefinition } from '../../types';
import { CATEGORY_COLORS, DIFFICULTY_COLORS } from '../../styles/theme';
import styles from './NodePalette.module.css';

const CATEGORY_ORDER = ['Data', 'IO', 'CNN', 'Normalization', 'RNN', 'Transformer', 'RL', 'Training', 'Tensor Operations', 'Control', 'Utility'];

// ── Operation Node Item ──

interface NodeItemProps {
  definition: NodeDefinition;
}

function NodeItem({ definition }: NodeItemProps) {
  const [hovered, setHovered] = useState(false);
  const { tn } = useI18n();

  const desc = tn(definition.node_name, 'description', definition.description);

  const handleDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData('application/codefyui-node', definition.node_name);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={desc}
      className={styles.nodeItem}
      style={{
        background: hovered ? '#2a2a2a' : 'transparent',
        borderColor: hovered ? '#444' : 'transparent',
      }}
    >
      <div className={styles.nodeItemName}>
        {definition.node_name}
      </div>
      {desc && (
        <div className={styles.nodeItemDesc}>
          {desc}
        </div>
      )}
    </div>
  );
}

// ── Preset Item ──

interface PresetItemProps {
  preset: PresetDefinition;
}

function PresetItem({ preset }: PresetItemProps) {
  const [hovered, setHovered] = useState(false);
  const difficulty = preset.tags.find((t) => t in DIFFICULTY_COLORS) ?? 'beginner';
  const difficultyColor = DIFFICULTY_COLORS[difficulty];

  const handleDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData('application/codefyui-preset', preset.preset_name);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={preset.description}
      className={styles.presetItem}
      style={{
        background: hovered ? 'rgba(212,160,23,0.08)' : 'transparent',
        borderColor: hovered ? 'rgba(212,160,23,0.3)' : 'transparent',
      }}
    >
      <div className={styles.presetHeader}>
        <div className={styles.presetName}>
          {preset.preset_name}
        </div>
        <span
          className={styles.presetDifficultyBadge}
          style={{
            background: `${difficultyColor}22`,
            color: difficultyColor,
          }}
        >
          {difficulty}
        </span>
      </div>
      <div className={styles.presetDesc}>
        {preset.description}
      </div>
      <div className={styles.presetNodeCount}>
        {preset.nodes.length} nodes
      </div>
    </div>
  );
}

// ── Category Section ──

interface CategorySectionProps {
  category: string;
  nodes: NodeDefinition[];
}

function CategorySection({ category, nodes }: CategorySectionProps) {
  const [expanded, setExpanded] = useState(true);
  const color = CATEGORY_COLORS[category] ?? '#607D8B';

  return (
    <div className={styles.categorySection}>
      <button
        onClick={() => setExpanded((prev) => !prev)}
        className={styles.categoryButton}
        style={{ borderBottom: `2px solid ${color}` }}
      >
        <span
          className={styles.categoryDot}
          style={{ background: color }}
        />
        <span className={styles.categoryName} style={{ color }}>
          {category}
        </span>
        <span className={styles.categoryCount}>
          {nodes.length}
        </span>
        <span className={styles.categoryChevron}>
          {expanded ? '▾' : '▸'}
        </span>
      </button>

      {expanded && (
        <div className={styles.categoryNodes}>
          {nodes.map((def) => (
            <NodeItem key={def.node_name} definition={def} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Preset Category Section ──

interface PresetCategorySectionProps {
  category: string;
  presets: PresetDefinition[];
}

function PresetCategorySection({ category, presets }: PresetCategorySectionProps) {
  const [expanded, setExpanded] = useState(true);
  const color = CATEGORY_COLORS[category] ?? '#D4A017';

  return (
    <div className={styles.categorySection}>
      <button
        onClick={() => setExpanded((prev) => !prev)}
        className={styles.categoryButton}
        style={{ borderBottom: `2px solid ${color}` }}
      >
        <span
          className={styles.categoryDot}
          style={{ background: color }}
        />
        <span className={styles.categoryName} style={{ color }}>
          {category}
        </span>
        <span className={styles.categoryCount}>
          {presets.length}
        </span>
        <span className={styles.categoryChevron}>
          {expanded ? '▾' : '▸'}
        </span>
      </button>

      {expanded && (
        <div className={styles.categoryNodes}>
          {presets.map((p) => (
            <PresetItem key={p.preset_name} preset={p} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main Palette ──

type PaletteTab = 'presets' | 'operations';

export function NodePalette() {
  const { categorized, loading, error, refetch } = useNodeDefinitions();
  const presetCategorized = useNodeDefStore((s) => s.presetCategorized);
  const presets = useNodeDefStore((s) => s.presets);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<PaletteTab>('presets');
  const { t } = useI18n();

  // Operations tab filtering
  const orderedCategories = [
    ...CATEGORY_ORDER.filter((c) => c in categorized),
    ...Object.keys(categorized).filter((c) => !CATEGORY_ORDER.includes(c)),
  ];

  const filteredCategorized: Record<string, NodeDefinition[]> = {};
  if (searchQuery.trim()) {
    const q = searchQuery.toLowerCase();
    for (const [cat, nodes] of Object.entries(categorized)) {
      const matched = nodes.filter(
        (n) =>
          n.node_name.toLowerCase().includes(q) ||
          n.description.toLowerCase().includes(q)
      );
      if (matched.length > 0) filteredCategorized[cat] = matched;
    }
  } else {
    Object.assign(filteredCategorized, categorized);
  }
  const displayCategories = orderedCategories.filter((c) => c in filteredCategorized);

  // Presets tab filtering
  const presetCategories = Object.keys(presetCategorized);
  const filteredPresets: Record<string, PresetDefinition[]> = {};
  if (searchQuery.trim()) {
    const q = searchQuery.toLowerCase();
    for (const [cat, ps] of Object.entries(presetCategorized)) {
      const matched = ps.filter(
        (p) =>
          p.preset_name.toLowerCase().includes(q) ||
          p.description.toLowerCase().includes(q) ||
          p.tags.some((tag) => tag.toLowerCase().includes(q))
      );
      if (matched.length > 0) filteredPresets[cat] = matched;
    }
  } else {
    Object.assign(filteredPresets, presetCategorized);
  }
  const displayPresetCategories = presetCategories.filter((c) => c in filteredPresets);

  return (
    <div className={styles.sidebar}>
      {/* Sidebar header */}
      <div className={styles.header}>
        <div className={styles.headerTitle}>
          {t('palette.title')}
        </div>
        <input
          type="text"
          placeholder={t('palette.search')}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className={styles.searchInput}
        />
      </div>

      {/* Tab bar */}
      <div className={styles.tabBar}>
        <button
          onClick={() => setActiveTab('presets')}
          className={styles.tabButton}
          style={{
            borderBottom: activeTab === 'presets' ? '2px solid #D4A017' : '2px solid transparent',
            color: activeTab === 'presets' ? '#D4A017' : '#666',
          }}
        >
          {t('palette.tabPresets')}
        </button>
        <button
          onClick={() => setActiveTab('operations')}
          className={styles.tabButton}
          style={{
            borderBottom: activeTab === 'operations' ? '2px solid #888' : '2px solid transparent',
            color: activeTab === 'operations' ? '#ccc' : '#666',
          }}
        >
          {t('palette.tabOperations')}
        </button>
      </div>

      {/* Content */}
      <div className={styles.content}>
        {loading && (
          <div className={styles.stateMessage}>
            {t('palette.loading')}
          </div>
        )}

        {error && (
          <div className={styles.errorWrapper}>
            <div className={styles.errorText}>
              {t('palette.loadFail', { error })}
            </div>
            <button onClick={refetch} className={styles.retryButton}>
              {t('palette.retry')}
            </button>
          </div>
        )}

        {/* Presets tab */}
        {!loading && !error && activeTab === 'presets' && (
          <>
            {presets.length === 0 && (
              <div className={styles.stateMessageMuted}>
                {t('palette.noPresets')}
              </div>
            )}
            {displayPresetCategories.length === 0 && presets.length > 0 && searchQuery && (
              <div className={styles.stateMessageMuted}>
                {t('palette.noMatch')}
              </div>
            )}
            {displayPresetCategories.map((cat) => (
              <PresetCategorySection
                key={cat}
                category={cat}
                presets={filteredPresets[cat]}
              />
            ))}
          </>
        )}

        {/* Operations tab */}
        {!loading && !error && activeTab === 'operations' && (
          <>
            {displayCategories.length === 0 && (
              <div className={styles.stateMessageMuted}>
                {searchQuery ? t('palette.noMatch') : t('palette.empty')}
              </div>
            )}
            {displayCategories.map((category) => (
              <CategorySection
                key={category}
                category={category}
                nodes={filteredCategorized[category]}
              />
            ))}
          </>
        )}
      </div>

      {/* Footer hint */}
      <div className={styles.footer}>
        {t('palette.hint')}
      </div>
    </div>
  );
}
