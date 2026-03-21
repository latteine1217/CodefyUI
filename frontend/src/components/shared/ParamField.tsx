import type { ParamDefinition } from '../../types';
import styles from './ParamField.module.css';

interface ParamFieldProps {
  param: ParamDefinition;
  value: any;
  onChange: (name: string, value: any) => void;
  label?: string;
}

export function ParamField({ param, value, onChange, label }: ParamFieldProps) {
  const displayLabel = label ?? param.name;

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

  if (param.param_type === 'int') {
    return (
      <div>
        <label className={styles.label}>{displayLabel}</label>
        <input
          type="number"
          value={value ?? param.default ?? 0}
          min={param.min_value ?? undefined}
          max={param.max_value ?? undefined}
          step={1}
          onChange={(e) => onChange(param.name, parseInt(e.target.value, 10))}
          className={styles.input}
        />
      </div>
    );
  }

  if (param.param_type === 'float') {
    return (
      <div>
        <label className={styles.label}>{displayLabel}</label>
        <input
          type="number"
          value={value ?? param.default ?? 0}
          min={param.min_value ?? undefined}
          max={param.max_value ?? undefined}
          step="any"
          onChange={(e) => onChange(param.name, parseFloat(e.target.value))}
          className={styles.input}
        />
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
