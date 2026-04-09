/**
 * Maps raw Python exception strings to beginner-friendly error messages.
 */
export function friendlyError(raw: string): string {
  // KeyError with specific key name
  const keyErrorMatch = raw.match(/KeyError:\s*'([^']+)'/);
  if (keyErrorMatch) {
    const key = keyErrorMatch[1];
    if (key === 'tensor') {
      return "This node expected a 'tensor' input but did not receive one. Check that all required inputs are connected.";
    }
    return `Missing required data: '${key}'. Ensure all inputs are connected.`;
  }

  // RuntimeError: shape mismatch
  if (/RuntimeError:.*shape\s+'[^']*'\s+is invalid/.test(raw)) {
    return 'Tensor shape mismatch. Check the dimensions of connected nodes.';
  }

  // ValueError: extract the message after "ValueError:"
  const valueErrorMatch = raw.match(/ValueError:\s*(.*)/);
  if (valueErrorMatch) {
    return valueErrorMatch[1].trim();
  }

  // Default: return raw message unchanged
  return raw;
}
