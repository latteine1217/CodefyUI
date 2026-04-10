import styles from './MigrationModal.module.css';

export interface MigrationModalProps {
  open: boolean;
  onAutoMark: () => void;
  onOpenAsDraft: () => void;
  onCancel: () => void;
  t: (key: string) => string;
}

export function MigrationModal({ open, onAutoMark, onOpenAsDraft, onCancel, t }: MigrationModalProps) {
  if (!open) return null;
  return (
    <div className={styles.overlay} onClick={onCancel}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <h2 className={styles.title}>{t('migration.noEntryPoints.title')}</h2>
        <p className={styles.body}>{t('migration.noEntryPoints.body')}</p>
        <div className={styles.buttons}>
          <button className={`${styles.button} ${styles.buttonPrimary}`} onClick={onAutoMark}>
            {t('migration.autoMark')}
          </button>
          <button className={styles.button} onClick={onOpenAsDraft}>
            {t('migration.openAsDraft')}
          </button>
          <button className={styles.button} onClick={onCancel}>
            {t('migration.cancel')}
          </button>
        </div>
      </div>
    </div>
  );
}
