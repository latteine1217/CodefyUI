import { ViewportPortal, useViewport } from '@xyflow/react';
import { useMemo } from 'react';
import { useTabStore } from '../../store/tabStore';
import { computeSegmentNodes } from '../../utils/segmentPath';
import type { Node as FlowNode } from '@xyflow/react';
import type { NodeData } from '../../types';

const BUBBLE_PAD = 28;
const BUBBLE_RADIUS = 28;
const BUBBLE_FILL = 'rgba(255, 180, 80, 0.22)';
const BUBBLE_STROKE = 'rgba(255, 140, 0, 0.6)';
const BADGE_FILL = '#ff9500';

interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

function nodeBBox(n: FlowNode<NodeData>): BBox {
  const w = n.measured?.width ?? n.width ?? 200;
  const h = n.measured?.height ?? n.height ?? 80;
  return { x: n.position.x, y: n.position.y, w, h };
}

function unionBBox(boxes: BBox[]): BBox | null {
  if (boxes.length === 0) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const b of boxes) {
    if (b.x < minX) minX = b.x;
    if (b.y < minY) minY = b.y;
    if (b.x + b.w > maxX) maxX = b.x + b.w;
    if (b.y + b.h > maxY) maxY = b.y + b.h;
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

export function SegmentBubble() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const activeSegment = activeTab.activeSegment;
  const nodes = activeTab.nodes;
  const edges = activeTab.edges;
  const { zoom } = useViewport();

  const segmentNodeIds = useMemo(() => {
    if (!activeSegment) return new Set<string>();
    return computeSegmentNodes(activeSegment.headNodeId, activeSegment.tailNodeId, nodes, edges);
  }, [activeSegment, nodes, edges]);

  if (!activeSegment || segmentNodeIds.size === 0) return null;

  const segmentNodes = nodes.filter((n) => segmentNodeIds.has(n.id));
  const union = unionBBox(segmentNodes.map(nodeBBox));
  if (!union) return null;

  const headNode = nodes.find((n) => n.id === activeSegment.headNodeId);
  const tailNode = nodes.find((n) => n.id === activeSegment.tailNodeId);

  const rect = {
    x: union.x - BUBBLE_PAD,
    y: union.y - BUBBLE_PAD,
    w: union.w + BUBBLE_PAD * 2,
    h: union.h + BUBBLE_PAD * 2,
  };

  const stroke = 2 / zoom;

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
          zIndex: 0,
        }}
      >
        <rect
          x={rect.x}
          y={rect.y}
          width={rect.w}
          height={rect.h}
          rx={BUBBLE_RADIUS}
          ry={BUBBLE_RADIUS}
          fill={BUBBLE_FILL}
          stroke={BUBBLE_STROKE}
          strokeWidth={stroke}
        />
        {headNode && <Badge box={nodeBBox(headNode)} anchor="top-left" text="HEAD" />}
        {tailNode && <Badge box={nodeBBox(tailNode)} anchor="bottom-right" text="TAIL" />}
      </svg>
    </ViewportPortal>
  );
}

interface BadgeProps {
  box: BBox;
  anchor: 'top-left' | 'bottom-right';
  text: string;
}

function Badge({ box, anchor, text }: BadgeProps) {
  const w = text.length * 8 + 14;
  const h = 18;
  const x = anchor === 'top-left' ? box.x - 4 : box.x + box.w - w + 4;
  const y = anchor === 'top-left' ? box.y - h - 4 : box.y + box.h + 4;
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={4}
        ry={4}
        fill={BADGE_FILL}
      />
      <text
        x={x + w / 2}
        y={y + h / 2 + 4}
        textAnchor="middle"
        fill="#ffffff"
        fontSize={11}
        fontWeight={700}
        fontFamily="ui-monospace, SFMono-Regular, monospace"
        style={{ letterSpacing: '0.06em' }}
      >
        {text}
      </text>
    </g>
  );
}
