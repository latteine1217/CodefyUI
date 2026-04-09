import { useState, useEffect, useCallback } from 'react';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { resolveSerializedNodes, resolveSerializedEdges } from '../../utils';
import { listExamples, loadExample } from '../../api/rest';
import type { ExampleSummary } from '../../api/rest';
import { useToastStore } from '../../store/toastStore';
import styles from './EmptyCanvasOverlay.module.css';

const CURATED_PATHS = [
  'Usage_Example/CNN-MNIST/TrainCNN-MNIST',
  'Usage_Example/CNN-MNIST/InferenceCNN-MNIST',
  'Model_Architecture/ResNet-SkipConnection-CNN',
  'Model_Architecture/GPT-DecoderOnly-Transformer',
];

const EXAMPLE_CATEGORY_COLORS: Record<string, string> = {
  Usage_Example: '#4CAF50',
  Model_Architecture: '#2196F3',
};

export function EmptyCanvasOverlay() {
  const setNodes = useTabStore((s) => s.setNodes);
  const setEdges = useTabStore((s) => s.setEdges);
  const { t } = useI18n();
  const addToast = useToastStore((s) => s.addToast);

  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listExamples()
      .then((all) => {
        // Pick curated examples in order, fallback to first 3
        const curated: ExampleSummary[] = [];
        for (const path of CURATED_PATHS) {
          const found = all.find((e) => e.path.replace(/\\/g, '/') === path);
          if (found) curated.push(found);
        }
        setExamples(curated.length > 0 ? curated : all.slice(0, 3));
      })
      .catch(() => setExamples([]))
      .finally(() => setLoading(false));
  }, []);

  const handleClick = useCallback(
    async (example: ExampleSummary) => {
      try {
        const data = await loadExample(example.path);
        const rawNodes = data.nodes ?? [];
        const edges = data.edges ?? [];

        const store = useNodeDefStore.getState();
        const importedPresets = Array.isArray(data.presets) ? data.presets : [];
        const mergedPresets = [...store.presets];
        for (const p of importedPresets) {
          if (!mergedPresets.some((ep) => ep.preset_name === p.preset_name)) {
            mergedPresets.push(p);
          }
        }

        setNodes(resolveSerializedNodes(rawNodes, store.definitions, mergedPresets));
        setEdges(resolveSerializedEdges(edges));

        if (importedPresets.length > 0) {
          useNodeDefStore.setState({ presets: mergedPresets });
        }
      } catch {
        addToast(t('empty.loadError'), 'error');
      }
    },
    [setNodes, setEdges, t],
  );

  return (
    <div className={styles.overlay}>
      <div className={styles.inner}>
        <div className={styles.title}>{t('empty.title')}</div>
        <div className={styles.subtitle}>{t('empty.subtitle')}</div>

        {loading && (
          <div className={styles.hint}>{t('empty.loading')}</div>
        )}

        {!loading && examples.length > 0 && (
          <div className={styles.quickStartGrid}>
            {examples.map((example) => {
              const catColor = EXAMPLE_CATEGORY_COLORS[example.category] ?? '#FF9800';
              const catLabel = example.category.replace(/_/g, ' ');
              return (
                <button
                  key={example.path}
                  onClick={() => handleClick(example)}
                  className={styles.presetCard}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = '#D4A017';
                    e.currentTarget.style.boxShadow = '0 4px 16px rgba(212,160,23,0.15)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = '#3a3a3a';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  <div className={styles.presetCardHeader}>
                    <span className={styles.presetCardName}>{example.name}</span>
                  </div>
                  <div className={styles.presetCardDesc}>
                    {example.description.length > 80
                      ? example.description.slice(0, 80) + '...'
                      : example.description}
                  </div>
                  <div className={styles.presetCardFooter}>
                    <span
                      className={styles.difficultyBadge}
                      style={{
                        background: `${catColor}22`,
                        color: catColor,
                      }}
                    >
                      {catLabel}
                    </span>
                    <span className={styles.nodeCount}>
                      {example.node_count} nodes
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}

        <div className={styles.hint}>{t('empty.hint')}</div>
      </div>
    </div>
  );
}
