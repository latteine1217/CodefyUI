import { useState } from 'react';
import { useNodeDefinitions } from '../../hooks/useNodeDefinitions';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useI18n } from '../../i18n';
import type { NodeDefinition, PresetDefinition } from '../../types';

const CATEGORY_COLORS: Record<string, string> = {
  CNN: '#4CAF50',
  RNN: '#2196F3',
  Transformer: '#9C27B0',
  RL: '#FF9800',
  Data: '#00BCD4',
  Training: '#F44336',
  IO: '#795548',
  Control: '#FF6F00',
  Utility: '#607D8B',
};

const CATEGORY_ORDER = ['Data', 'IO', 'CNN', 'RNN', 'Transformer', 'RL', 'Training', 'Control', 'Utility'];

const DIFFICULTY_COLORS: Record<string, string> = {
  beginner: '#4CAF50',
  intermediate: '#FF9800',
  advanced: '#F44336',
};

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
      style={{
        padding: '6px 10px',
        margin: '2px 6px',
        borderRadius: 5,
        cursor: 'grab',
        background: hovered ? '#2a2a2a' : 'transparent',
        border: '1px solid',
        borderColor: hovered ? '#444' : 'transparent',
        transition: 'all 0.15s',
        userSelect: 'none',
      }}
    >
      <div
        style={{
          fontSize: '0.8125rem',
          fontWeight: 500,
          color: '#ddd',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {definition.node_name}
      </div>
      {desc && (
        <div
          style={{
            fontSize: '0.6875rem',
            color: '#777',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            marginTop: 1,
          }}
        >
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
      style={{
        padding: '8px 10px',
        margin: '3px 6px',
        borderRadius: 6,
        cursor: 'grab',
        background: hovered ? 'rgba(212,160,23,0.08)' : 'transparent',
        border: '1px solid',
        borderColor: hovered ? 'rgba(212,160,23,0.3)' : 'transparent',
        transition: 'all 0.15s',
        userSelect: 'none',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
        <div
          style={{
            fontSize: '0.8125rem',
            fontWeight: 600,
            color: '#ddd',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            flex: 1,
          }}
        >
          {preset.preset_name}
        </div>
        <span
          style={{
            fontSize: '0.5625rem',
            padding: '1px 4px',
            borderRadius: 3,
            background: `${DIFFICULTY_COLORS[difficulty]}22`,
            color: DIFFICULTY_COLORS[difficulty],
            fontWeight: 600,
            flexShrink: 0,
          }}
        >
          {difficulty}
        </span>
      </div>
      <div
        style={{
          fontSize: '0.6875rem',
          color: '#777',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {preset.description}
      </div>
      <div style={{ fontSize: '0.625rem', color: '#555', marginTop: 3 }}>
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
    <div style={{ marginBottom: 4 }}>
      <button
        onClick={() => setExpanded((prev) => !prev)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 10px',
          background: 'transparent',
          border: 'none',
          borderBottom: `2px solid ${color}`,
          cursor: 'pointer',
          color: '#eee',
          textAlign: 'left',
        }}
      >
        <span
          style={{
            display: 'inline-block',
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: color,
            flexShrink: 0,
          }}
        />
        <span style={{ fontSize: '0.8125rem', fontWeight: 700, letterSpacing: '0.05em', flex: 1, color }}>
          {category}
        </span>
        <span style={{ fontSize: '0.6875rem', color: '#666' }}>
          {nodes.length}
        </span>
        <span style={{ fontSize: '0.6875rem', color: '#555', marginLeft: 4 }}>
          {expanded ? '▾' : '▸'}
        </span>
      </button>

      {expanded && (
        <div style={{ paddingTop: 2, paddingBottom: 4 }}>
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
    <div style={{ marginBottom: 4 }}>
      <button
        onClick={() => setExpanded((prev) => !prev)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 10px',
          background: 'transparent',
          border: 'none',
          borderBottom: `2px solid ${color}`,
          cursor: 'pointer',
          color: '#eee',
          textAlign: 'left',
        }}
      >
        <span
          style={{
            display: 'inline-block',
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: color,
            flexShrink: 0,
          }}
        />
        <span style={{ fontSize: '0.8125rem', fontWeight: 700, letterSpacing: '0.05em', flex: 1, color }}>
          {category}
        </span>
        <span style={{ fontSize: '0.6875rem', color: '#666' }}>
          {presets.length}
        </span>
        <span style={{ fontSize: '0.6875rem', color: '#555', marginLeft: 4 }}>
          {expanded ? '▾' : '▸'}
        </span>
      </button>

      {expanded && (
        <div style={{ paddingTop: 2, paddingBottom: 4 }}>
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
    <div
      style={{
        width: 250,
        height: '100%',
        background: '#161616',
        borderRight: '1px solid #2a2a2a',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
      }}
    >
      {/* Sidebar header */}
      <div
        style={{
          padding: '12px 10px 8px',
          borderBottom: '1px solid #2a2a2a',
          flexShrink: 0,
        }}
      >
        <div
          style={{
            fontSize: '0.75rem',
            fontWeight: 700,
            color: '#888',
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            marginBottom: 8,
          }}
        >
          {t('palette.title')}
        </div>
        <input
          type="text"
          placeholder={t('palette.search')}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            width: '100%',
            padding: '5px 8px',
            background: '#222',
            border: '1px solid #333',
            borderRadius: 4,
            color: '#ddd',
            fontSize: '0.8125rem',
            outline: 'none',
            boxSizing: 'border-box',
          }}
        />
      </div>

      {/* Tab bar */}
      <div
        style={{
          display: 'flex',
          borderBottom: '1px solid #2a2a2a',
          flexShrink: 0,
        }}
      >
        <button
          onClick={() => setActiveTab('presets')}
          style={{
            flex: 1,
            padding: '7px 0',
            background: 'transparent',
            border: 'none',
            borderBottom: activeTab === 'presets' ? '2px solid #D4A017' : '2px solid transparent',
            color: activeTab === 'presets' ? '#D4A017' : '#666',
            fontSize: '0.75rem',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.15s',
          }}
        >
          {t('palette.tabPresets')}
        </button>
        <button
          onClick={() => setActiveTab('operations')}
          style={{
            flex: 1,
            padding: '7px 0',
            background: 'transparent',
            border: 'none',
            borderBottom: activeTab === 'operations' ? '2px solid #888' : '2px solid transparent',
            color: activeTab === 'operations' ? '#ccc' : '#666',
            fontSize: '0.75rem',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.15s',
          }}
        >
          {t('palette.tabOperations')}
        </button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', paddingTop: 4 }}>
        {loading && (
          <div
            style={{
              padding: '20px 10px',
              textAlign: 'center',
              color: '#666',
              fontSize: '0.8125rem',
            }}
          >
            {t('palette.loading')}
          </div>
        )}

        {error && (
          <div style={{ padding: '12px 10px' }}>
            <div
              style={{
                fontSize: '0.75rem',
                color: '#F44336',
                marginBottom: 8,
                lineHeight: 1.4,
              }}
            >
              {t('palette.loadFail', { error })}
            </div>
            <button
              onClick={refetch}
              style={{
                fontSize: '0.75rem',
                padding: '4px 8px',
                background: '#333',
                border: '1px solid #444',
                borderRadius: 4,
                color: '#ccc',
                cursor: 'pointer',
              }}
            >
              {t('palette.retry')}
            </button>
          </div>
        )}

        {/* Presets tab */}
        {!loading && !error && activeTab === 'presets' && (
          <>
            {presets.length === 0 && (
              <div
                style={{
                  padding: '20px 10px',
                  textAlign: 'center',
                  color: '#555',
                  fontSize: '0.8125rem',
                }}
              >
                {t('palette.noPresets')}
              </div>
            )}
            {displayPresetCategories.length === 0 && presets.length > 0 && searchQuery && (
              <div
                style={{
                  padding: '20px 10px',
                  textAlign: 'center',
                  color: '#555',
                  fontSize: '0.8125rem',
                }}
              >
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
              <div
                style={{
                  padding: '20px 10px',
                  textAlign: 'center',
                  color: '#555',
                  fontSize: '0.8125rem',
                }}
              >
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
      <div
        style={{
          padding: '6px 10px',
          borderTop: '1px solid #2a2a2a',
          fontSize: '0.6875rem',
          color: '#444',
          textAlign: 'center',
          flexShrink: 0,
        }}
      >
        {t('palette.hint')}
      </div>
    </div>
  );
}
