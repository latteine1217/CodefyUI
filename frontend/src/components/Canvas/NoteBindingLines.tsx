import { ViewportPortal, useViewport } from '@xyflow/react';
import { useTabStore } from '../../store/tabStore';

/** Find the point where a line from `center` toward `target` exits the rectangle. */
function rectEdgePoint(
  cx: number, cy: number, w: number, h: number,
  tx: number, ty: number,
): { x: number; y: number } {
  const dx = tx - cx;
  const dy = ty - cy;
  if (dx === 0 && dy === 0) return { x: cx, y: cy };

  const hw = w / 2;
  const hh = h / 2;
  // Scale factor to reach each edge
  const sx = dx !== 0 ? hw / Math.abs(dx) : Infinity;
  const sy = dy !== 0 ? hh / Math.abs(dy) : Infinity;
  const s = Math.min(sx, sy);

  return { x: cx + dx * s, y: cy + dy * s };
}

export function NoteBindingLines() {
  const nodes = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    return tab?.nodes ?? [];
  });
  const { zoom } = useViewport();

  const lines: { id: string; x1: number; y1: number; x2: number; y2: number }[] = [];

  for (const n of nodes) {
    if (n.type !== 'noteNode' || !n.data.boundToNodeId) continue;
    const parent = nodes.find((p) => p.id === n.data.boundToNodeId);
    if (!parent) continue;

    const nw = n.measured?.width ?? n.width ?? 200;
    const nh = n.measured?.height ?? n.height ?? 60;
    const pw = parent.measured?.width ?? parent.width ?? 200;
    const ph = parent.measured?.height ?? parent.height ?? 80;

    const ncx = n.position.x + nw / 2;
    const ncy = n.position.y + nh / 2;
    const pcx = parent.position.x + pw / 2;
    const pcy = parent.position.y + ph / 2;

    const from = rectEdgePoint(ncx, ncy, nw, nh, pcx, pcy);
    const to = rectEdgePoint(pcx, pcy, pw, ph, ncx, ncy);

    lines.push({ id: n.id, x1: from.x, y1: from.y, x2: to.x, y2: to.y });
  }

  if (lines.length === 0) return null;

  const sw = 1 / zoom; // consistent stroke width across zoom levels
  const dash = 4 / zoom;

  return (
    <ViewportPortal>
      <svg
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          overflow: 'visible',
          pointerEvents: 'none',
        }}
      >
        {lines.map((l) => (
          <line
            key={l.id}
            x1={l.x1}
            y1={l.y1}
            x2={l.x2}
            y2={l.y2}
            stroke="#666"
            strokeWidth={sw}
            strokeDasharray={`${dash} ${dash}`}
            opacity={0.6}
          />
        ))}
      </svg>
    </ViewportPortal>
  );
}
