import {
  SKELETON_EDGES,
  type Pose,
} from "../pose/keypoints";
import { DEFAULT_MIN_KEYPOINT_SCORE } from "../pose/normalize";

export interface DrawOptions {
  minScore?: number;
  pointColor?: string;
  edgeColor?: string;
  pointRadius?: number;
  lineWidth?: number;
  /** Bone edges (keypoint index pairs) to draw in the highlight color. */
  highlightEdges?: Array<[number, number]>;
  /** Color for highlighted bones (e.g. the worst-diverging limb). */
  highlightColor?: string;
}

/** Order-independent key for an edge between keypoints a and b. */
function edgeKey(a: number, b: number): string {
  return a < b ? `${a}-${b}` : `${b}-${a}`;
}

/**
 * Linear map from source-video pixel space into the target canvas:
 *   canvasX = x * sx + dx,  canvasY = y * sy + dy
 * Identity (the default) draws 1:1, as the live overlays do.
 */
export interface PoseTransform {
  sx: number;
  sy: number;
  dx: number;
  dy: number;
}

const IDENTITY: PoseTransform = { sx: 1, sy: 1, dx: 0, dy: 0 };

/**
 * Draw a pose's bones and joints into an existing 2D context under a coordinate
 * transform, *without* clearing. Shared by the live overlays (identity) and the
 * export composite (scaled/offset into a sub-region). Caller controls clearing.
 */
export function strokePose(
  ctx: CanvasRenderingContext2D,
  pose: Pose,
  opts: DrawOptions & { transform?: PoseTransform } = {},
): void {
  const t = opts.transform ?? IDENTITY;
  const minScore = opts.minScore ?? DEFAULT_MIN_KEYPOINT_SCORE;
  const pointColor = opts.pointColor ?? "#00e5ff";
  const edgeColor = opts.edgeColor ?? "#00ff9c";
  const pointRadius = opts.pointRadius ?? 4;
  const lineWidth = opts.lineWidth ?? 2;
  const highlightColor = opts.highlightColor ?? "#ff3b30";
  const highlight = new Set(
    (opts.highlightEdges ?? []).map(([a, b]) => edgeKey(a, b)),
  );
  const tx = (x: number) => x * t.sx + t.dx;
  const ty = (y: number) => y * t.sy + t.dy;

  const kp = pose.keypoints;

  // Bones first so joints sit on top. Highlighted limbs draw thicker and in the
  // highlight color so the worst-diverging body part stands out.
  ctx.lineCap = "round";
  for (const [a, b] of SKELETON_EDGES) {
    const ka = kp[a];
    const kb = kp[b];
    if (!ka || !kb) continue;
    if ((ka.score ?? 0) < minScore || (kb.score ?? 0) < minScore) continue;
    const isHi = highlight.has(edgeKey(a, b));
    ctx.strokeStyle = isHi ? highlightColor : edgeColor;
    ctx.lineWidth = isHi ? lineWidth * 1.7 : lineWidth;
    ctx.beginPath();
    ctx.moveTo(tx(ka.x), ty(ka.y));
    ctx.lineTo(tx(kb.x), ty(kb.y));
    ctx.stroke();
  }

  // Joints.
  ctx.fillStyle = pointColor;
  for (const k of kp) {
    if ((k.score ?? 0) < minScore) continue;
    ctx.beginPath();
    ctx.arc(tx(k.x), ty(k.y), pointRadius, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * Resize a canvas's drawing buffer to match a video's intrinsic resolution so
 * skeleton coordinates (which are in source-video pixels) line up exactly.
 * Returns true if the canvas was resized.
 */
export function syncCanvasToVideo(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
): boolean {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return false;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  return false;
}

export function clearCanvas(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext("2d");
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Draw a pose's bones and keypoints onto a canvas. Coordinates are assumed to
 * be in the source video's pixel space (matching the canvas buffer size).
 */
export function drawSkeleton(
  canvas: HTMLCanvasElement,
  pose: Pose,
  opts: DrawOptions = {},
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Live overlays draw 1:1 in source-video pixels; size strokes to the canvas.
  strokePose(ctx, pose, {
    pointRadius: Math.max(3, canvas.width * 0.006),
    lineWidth: Math.max(2, canvas.width * 0.004),
    ...opts,
  });
}

/** One skeleton to draw in a multi-person overlay. */
export interface SkeletonLayer {
  pose: Pose;
  /** Draw faded (a non-selected bystander). */
  dim?: boolean;
  /** Bone edges to highlight (e.g. the worst-diverging limb on the locked one). */
  highlightEdges?: Array<[number, number]>;
}

/**
 * Draw several skeletons onto one canvas in a single clear: bystanders faded,
 * the selected dancer at full strength. Used by multi-person mode so the user
 * can see everyone the detector found and which one is locked.
 */
export function drawSkeletons(
  canvas: HTMLCanvasElement,
  layers: SkeletonLayer[],
  opts: DrawOptions = {},
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const base = {
    pointRadius: Math.max(3, canvas.width * 0.006),
    lineWidth: Math.max(2, canvas.width * 0.004),
    ...opts,
  };
  for (const layer of layers) {
    ctx.save();
    if (layer.dim) {
      ctx.globalAlpha = 0.35;
      strokePose(ctx, layer.pose, {
        ...base,
        // Muted gray bystander so the locked dancer's accent colors stand out.
        pointColor: "#9aa0a6",
        edgeColor: "#9aa0a6",
        highlightEdges: undefined,
      });
    } else {
      strokePose(ctx, layer.pose, {
        ...base,
        highlightEdges: layer.highlightEdges,
      });
    }
    ctx.restore();
  }
}
