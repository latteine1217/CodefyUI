import type { ParamDefinition } from '../../types';

interface ParamFieldProps {
  param: ParamDefinition;
  value: any;
  onChange: (name: string, value: any) => void;
  label?: string;
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '5px 8px',
  background: '#222',
  border: '1px solid #444',
  borderRadius: 4,
  color: '#ddd',
  fontSize: '0.8125rem',
  outline: 'none',
  boxSizing: 'border-box',
};

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: '0.75rem',
  color: '#999',
  marginBottom: 3,
  fontWeight: 500,
};

export function ParamField({ param, value, onChange, label }: ParamFieldProps) {
  const displayLabel = label ?? param.name;

  if (param.param_type === 'bool') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <input
          type="checkbox"
          id={`param-${param.name}`}
          checked={Boolean(value)}
          onChange={(e) => onChange(param.name, e.target.checked)}
          style={{ width: 14, height: 14, cursor: 'pointer', accentColor: '#2196F3' }}
        />
        <label
          htmlFor={`param-${param.name}`}
          style={{ ...labelStyle, marginBottom: 0, cursor: 'pointer', color: '#ccc' }}
        >
          {displayLabel}
        </label>
      </div>
    );
  }

  if (param.param_type === 'select') {
    return (
      <div>
        <label style={labelStyle}>{displayLabel}</label>
        <select
          value={value ?? param.default}
          onChange={(e) => onChange(param.name, e.target.value)}
          style={{
            ...inputStyle,
            cursor: 'pointer',
            appearance: 'none',
          }}
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
        <label style={labelStyle}>{displayLabel}</label>
        <input
          type="number"
          value={value ?? param.default ?? 0}
          min={param.min_value ?? undefined}
          max={param.max_value ?? undefined}
          step={1}
          onChange={(e) => onChange(param.name, parseInt(e.target.value, 10))}
          style={inputStyle}
        />
      </div>
    );
  }

  if (param.param_type === 'float') {
    return (
      <div>
        <label style={labelStyle}>{displayLabel}</label>
        <input
          type="number"
          value={value ?? param.default ?? 0}
          min={param.min_value ?? undefined}
          max={param.max_value ?? undefined}
          step="any"
          onChange={(e) => onChange(param.name, parseFloat(e.target.value))}
          style={inputStyle}
        />
      </div>
    );
  }

  // Default: string
  return (
    <div>
      <label style={labelStyle}>{displayLabel}</label>
      <input
        type="text"
        value={value ?? param.default ?? ''}
        onChange={(e) => onChange(param.name, e.target.value)}
        style={inputStyle}
      />
    </div>
  );
}
