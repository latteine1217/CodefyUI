import {
  BaseEdge,
  getBezierPath,
  getSmoothStepPath,
  Position,
  type EdgeProps,
} from '@xyflow/react';

const HORIZONTAL_SKIP_THRESHOLD = 380;
const VERTICAL_SKIP_THRESHOLD = 150;
const MINOR_TOLERANCE = 80;
const ROW_TRANSITION_DY_THRESHOLD = 200;
const COL_TRANSITION_DX_THRESHOLD = 200;
const ARC_OFFSET_BASE = 100;
const ARC_OFFSET_MAX_EXTRA = 120;
const ARC_OFFSET_SCALE = 0.15;
const PULL_OUT = 50;
const MINOR_FLAT_EPSILON = 20;
const ARC_JITTER_BUCKETS = 4;
const ARC_JITTER_STEP = 28;
const SMOOTH_STEP_BORDER_RADIUS = 20;

function isHorizontalPosition(p: Position): boolean {
  return p === Position.Left || p === Position.Right;
}

function computeArcOffset(major: number): number {
  return ARC_OFFSET_BASE + Math.min(Math.abs(major) * ARC_OFFSET_SCALE, ARC_OFFSET_MAX_EXTRA);
}

function computeArcDirection(minor: number): number {
  if (Math.abs(minor) < MINOR_FLAT_EPSILON) return -1;
  return Math.sign(minor);
}

function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

function buildSkipPath(
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
  horizontal: boolean,
  arcDir: number,
  arcOffset: number,
): string {
  if (horizontal) {
    const c1x = sourceX + PULL_OUT;
    const c1y = sourceY + arcDir * arcOffset;
    const c2x = targetX - PULL_OUT;
    const c2y = targetY + arcDir * arcOffset;
    return `M ${sourceX},${sourceY} C ${c1x},${c1y} ${c2x},${c2y} ${targetX},${targetY}`;
  }
  const c1x = sourceX + arcDir * arcOffset;
  const c1y = sourceY + PULL_OUT;
  const c2x = targetX + arcDir * arcOffset;
  const c2y = targetY - PULL_OUT;
  return `M ${sourceX},${sourceY} C ${c1x},${c1y} ${c2x},${c2y} ${targetX},${targetY}`;
}

export function SmartDataEdge(props: EdgeProps) {
  const {
    id,
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    style,
    markerEnd,
    interactionWidth,
  } = props;

  const horizontal = isHorizontalPosition(sourcePosition);
  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const major = horizontal ? dx : dy;
  const minor = horizontal ? dy : dx;

  const isRowTransition = horizontal
    ? Math.abs(dy) > ROW_TRANSITION_DY_THRESHOLD
    : Math.abs(dx) > COL_TRANSITION_DX_THRESHOLD;

  const majorThreshold = horizontal ? HORIZONTAL_SKIP_THRESHOLD : VERTICAL_SKIP_THRESHOLD;
  const isSkip =
    !isRowTransition &&
    Math.abs(major) > majorThreshold &&
    Math.abs(minor) < MINOR_TOLERANCE;

  let path: string;
  if (isRowTransition) {
    const [smoothStepPath] = getSmoothStepPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
      borderRadius: SMOOTH_STEP_BORDER_RADIUS,
    });
    path = smoothStepPath;
  } else if (isSkip) {
    const jitter = (hashString(id) % ARC_JITTER_BUCKETS) * ARC_JITTER_STEP;
    const arcOffset = computeArcOffset(major) + jitter;
    const arcDir = computeArcDirection(minor);
    path = buildSkipPath(sourceX, sourceY, targetX, targetY, horizontal, arcDir, arcOffset);
  } else {
    const [bezier] = getBezierPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
    });
    path = bezier;
  }

  return (
    <BaseEdge
      id={id}
      path={path}
      style={style}
      markerEnd={markerEnd}
      interactionWidth={interactionWidth}
    />
  );
}
