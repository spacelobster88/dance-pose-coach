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

  const minScore = opts.minScore ?? DEFAULT_MIN_KEYPOINT_SCORE;
  const pointColor = opts.pointColor ?? "#00e5ff";
  const edgeColor = opts.edgeColor ?? "#00ff9c";
  const pointRadius = opts.pointRadius ?? Math.max(3, canvas.width * 0.006);
  const lineWidth = opts.lineWidth ?? Math.max(2, canvas.width * 0.004);
  const highlightColor = opts.highlightColor ?? "#ff3b30";
  const highlight = new Set(
    (opts.highlightEdges ?? []).map(([a, b]) => edgeKey(a, b)),
  );

  ctx.clearRect(0, 0, canvas.width, canvas.height);

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
    ctx.moveTo(ka.x, ka.y);
    ctx.lineTo(kb.x, kb.y);
    ctx.stroke();
  }

  // Joints.
  ctx.fillStyle = pointColor;
  for (const k of kp) {
    if ((k.score ?? 0) < minScore) continue;
    ctx.beginPath();
    ctx.arc(k.x, k.y, pointRadius, 0, Math.PI * 2);
    ctx.fill();
  }
}
