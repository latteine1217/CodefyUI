import type { ConnectionLineComponentProps } from '@xyflow/react';

export function CustomConnectionLine({ fromX, fromY, toX, toY }: ConnectionLineComponentProps) {
  return (
    <g>
      <path
        fill="none"
        stroke="#888"
        strokeWidth={2}
        d={`M${fromX},${fromY} C${fromX + 80},${fromY} ${toX - 80},${toY} ${toX},${toY}`}
      />
      <circle cx={toX} cy={toY} r={4} fill="#888" />
    </g>
  );
}
