import { BaseEdge, getBezierPath, type EdgeProps } from '@xyflow/react';

export function TriggerEdge(props: EdgeProps) {
  const [path] = getBezierPath({
    sourceX: props.sourceX,
    sourceY: props.sourceY,
    sourcePosition: props.sourcePosition,
    targetX: props.targetX,
    targetY: props.targetY,
    targetPosition: props.targetPosition,
  });

  return (
    <BaseEdge
      id={props.id}
      path={path}
      style={{
        stroke: '#22c55e',
        strokeDasharray: '6 4',
        strokeWidth: 2,
      }}
    />
  );
}
