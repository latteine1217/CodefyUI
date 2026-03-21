import { describe, expect, it } from 'vitest';
import { generateId, getPortColor, isValidConnection, DATA_TYPE_COLORS } from './index';

describe('generateId', () => {
  it('returns a valid UUID string', () => {
    const id = generateId();
    expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
  });

  it('returns unique values', () => {
    const ids = new Set(Array.from({ length: 100 }, () => generateId()));
    expect(ids.size).toBe(100);
  });
});

describe('getPortColor', () => {
  it('returns the correct color for known types', () => {
    expect(getPortColor('TENSOR')).toBe('#4CAF50');
    expect(getPortColor('MODEL')).toBe('#2196F3');
    expect(getPortColor('DATASET')).toBe('#FF9800');
  });

  it('is case-insensitive', () => {
    expect(getPortColor('tensor')).toBe('#4CAF50');
    expect(getPortColor('Tensor')).toBe('#4CAF50');
  });

  it('returns ANY color for unknown types', () => {
    expect(getPortColor('UNKNOWN_TYPE')).toBe(DATA_TYPE_COLORS['ANY']);
  });
});

describe('isValidConnection', () => {
  it('allows same type connections', () => {
    expect(isValidConnection('TENSOR', 'TENSOR')).toBe(true);
    expect(isValidConnection('MODEL', 'MODEL')).toBe(true);
  });

  it('allows ANY on either side', () => {
    expect(isValidConnection('ANY', 'TENSOR')).toBe(true);
    expect(isValidConnection('TENSOR', 'ANY')).toBe(true);
    expect(isValidConnection('ANY', 'ANY')).toBe(true);
  });

  it('allows IMAGE to TENSOR', () => {
    expect(isValidConnection('IMAGE', 'TENSOR')).toBe(true);
  });

  it('rejects incompatible types', () => {
    expect(isValidConnection('TENSOR', 'MODEL')).toBe(false);
    expect(isValidConnection('MODEL', 'TENSOR')).toBe(false);
    expect(isValidConnection('OPTIMIZER', 'LOSS_FN')).toBe(false);
  });
});
