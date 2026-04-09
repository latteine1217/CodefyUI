import { useCallback, useEffect, useRef, useState } from 'react';
import type { ParamDefinition } from '../../types';
import { listModelFiles, uploadModelFile } from '../../api/rest';
import { useToastStore } from '../../store/toastStore';
import styles from './ParamField.module.css';

interface ParamFieldProps {
  param: ParamDefinition;
  value: any;
  onChange: (name: string, value: any) => void;
  label?: string;
}

function ModelFileField({
  param,
  value,
  onChange,
  displayLabel,
}: {
  param: ParamDefinition;
  value: any;
  onChange: (name: string, value: any) => void;
  displayLabel: string;
}) {
  const [files, setFiles] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    listModelFiles().then((list) => setFiles(list.map((f) => f.filename)));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadModelFile(file);
      refresh();
      onChange(param.name, result.filename);
    } catch (err: any) {
      useToastStore.getState().addToast(err.message ?? 'Upload failed', 'error');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div>
      <label className={styles.label}>{displayLabel}</label>
      <div className={styles.modelFileRow}>
        <select
          value={value ?? ''}
          onChange={(e) => onChange(param.name, e.target.value)}
          className={`${styles.input} ${styles.select} ${styles.modelFileSelect}`}
        >
          <option value="" style={{ background: '#222' }}>
            -- select file --
          </option>
          {files.map((f) => (
            <option key={f} value={f} style={{ background: '#222' }}>
              {f}
            </option>
          ))}
        </select>
        <button
          type="button"
          className={styles.modelFileBtn}
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? '...' : '↑'}
        </button>
        <button
          type="button"
          className={styles.modelFileBtn}
          onClick={refresh}
          title="Refresh file list"
        >
          ↻
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pt,.pth,.safetensors,.ckpt,.bin"
          style={{ display: 'none' }}
          onChange={handleUpload}
        />
      </div>
    </div>
  );
}

export function ParamField({ param, value, onChange, label }: ParamFieldProps) {
  const displayLabel = label ?? param.name;

  if (param.param_type === 'model_file') {
    return (
      <ModelFileField
        param={param}
        value={value}
        onChange={onChange}
        displayLabel={displayLabel}
      />
    );
  }

  if (param.param_type === 'bool') {
    return (
      <div className={styles.checkboxRow}>
        <input
          type="checkbox"
          id={`param-${param.name}`}
          checked={Boolean(value)}
          onChange={(e) => onChange(param.name, e.target.checked)}
          className={styles.checkbox}
        />
        <label htmlFor={`param-${param.name}`} className={styles.boolLabel}>
          {displayLabel}
        </label>
      </div>
    );
  }

  if (param.param_type === 'select') {
    return (
      <div>
        <label className={styles.label}>{displayLabel}</label>
        <select
          value={value ?? param.default}
          onChange={(e) => onChange(param.name, e.target.value)}
          className={`${styles.input} ${styles.select}`}
        >
          {param.options.map((opt) => (
            <option key={opt} value={opt} style={{ background: '#222' }}>
              {opt}
            </option>
          ))}
        </select>
      </div>
    );
  }

  if (param.param_type === 'int' || param.param_type === 'float') {
    const numVal = Number(value ?? param.default ?? 0);
    const hasMin = param.min_value != null;
    const hasMax = param.max_value != null;
    const outOfRange =
      !isNaN(numVal) &&
      ((hasMin && numVal < param.min_value!) || (hasMax && numVal > param.max_value!));
    const isInt = param.param_type === 'int';

    return (
      <div>
        <label className={styles.label}>{displayLabel}</label>
        <input
          type="number"
          value={value ?? param.default ?? 0}
          min={param.min_value ?? undefined}
          max={param.max_value ?? undefined}
          step={isInt ? 1 : 'any'}
          onChange={(e) =>
            onChange(param.name, isInt ? parseInt(e.target.value, 10) : parseFloat(e.target.value))
          }
          className={`${styles.input} ${outOfRange ? styles.inputError : ''}`}
        />
        {outOfRange && (hasMin || hasMax) && (
          <span className={styles.errorHint}>
            {hasMin && hasMax
              ? `Range: ${param.min_value} — ${param.max_value}`
              : hasMin
                ? `Min: ${param.min_value}`
                : `Max: ${param.max_value}`}
          </span>
        )}
      </div>
    );
  }

  // Default: string
  return (
    <div>
      <label className={styles.label}>{displayLabel}</label>
      <input
        type="text"
        value={value ?? param.default ?? ''}
        onChange={(e) => onChange(param.name, e.target.value)}
        className={styles.input}
      />
    </div>
  );
}
