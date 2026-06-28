import {
  SKELETON_EDGES,
  type Pose,
} from "../pose/keypoints";
import { DEFAULT_MIN_KEYPOINT_SCORE } from "../pose/normalize";
import type { TrackedPose } from "../pose/tracker";

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

/**
 * Multi-person overlay keyed on tracked ids: every candidate the detector found
 * is drawn faded except the locked dancer, who is drawn at full strength (with
 * any highlight edges). A thin convenience over `drawSkeletons` for the
 * live-webcam pick/hold path, so app.ts can pass the raw tracked array + the
 * locked id directly. `held`, when given, is the coasted pose to draw for the
 * locked dancer if it wasn't detected this frame (so a brief dropout doesn't
 * blink the lock out).
 */
export function drawMultiTarget(
  ctx: CanvasRenderingContext2D | HTMLCanvasElement,
  candidates: TrackedPose[],
  lockedId: number | null,
  opts: DrawOptions & {
    /** Highlight edges to apply to the locked dancer only. */
    highlightEdges?: Array<[number, number]>;
    /** A held/coasted pose for the locked dancer, drawn if it's not detected. */
    held?: Pose | null;
  } = {},
): void {
  const canvas =
    ctx instanceof HTMLCanvasElement ? ctx : (ctx.canvas as HTMLCanvasElement);
  const { highlightEdges, held, ...drawOpts } = opts;
  const layers: SkeletonLayer[] = candidates.map((t) => ({
    pose: t.pose,
    dim: t.id !== lockedId,
    highlightEdges: t.id === lockedId ? highlightEdges : undefined,
  }));
  // The locked dancer is coasting (not among this frame's detections): draw the
  // held pose so the lock stays visible.
  if (held && lockedId !== null && !candidates.some((t) => t.id === lockedId)) {
    layers.push({ pose: held, dim: false, highlightEdges });
  }
  if (layers.length) drawSkeletons(canvas, layers, drawOpts);
  else clearCanvas(canvas);
}

/** A display rectangle (subset of DOMRect) for coordinate mapping. */
export interface DisplayRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Map a pointer position in viewport/CSS pixels into source-video pixel space,
 * undoing both the `object-fit: contain` letterbox and an optional horizontal
 * (selfie) mirror.
 *
 * The video is rendered centered inside `rect` at the largest size that fits
 * while preserving its `videoW × videoH` aspect ratio, leaving letterbox bars on
 * the two narrow sides. A click in those bars isn't over any pixel of the video,
 * so this returns `null` (the caller should ignore it). When `mirrored` is true
 * the feed is flipped about its vertical center (CSS `transform: scaleX(-1)`),
 * so the recovered source X is mirrored back: `x → videoW - x`.
 *
 * Pure and DOM-free so it's unit-testable headless.
 */
export function mapDisplayToSource(
  clientX: number,
  clientY: number,
  rect: DisplayRect,
  videoW: number,
  videoH: number,
  mirrored: boolean,
): { x: number; y: number } | null {
  if (!rect.width || !rect.height || !videoW || !videoH) return null;
  // Largest scale that fits the video inside the box (object-fit: contain).
  const scale = Math.min(rect.width / videoW, rect.height / videoH);
  const drawnW = videoW * scale;
  const drawnH = videoH * scale;
  // Letterbox offsets that center the drawn video inside the box.
  const offX = (rect.width - drawnW) / 2;
  const offY = (rect.height - drawnH) / 2;
  // Position within the box, in box pixels.
  const bx = clientX - rect.left;
  const by = clientY - rect.top;
  // Reject clicks in the letterbox margin (not over the video content).
  if (bx < offX || bx > offX + drawnW || by < offY || by > offY + drawnH) {
    return null;
  }
  // Back into source-video pixels.
  let sx = (bx - offX) / scale;
  const sy = (by - offY) / scale;
  if (mirrored) sx = videoW - sx;
  return { x: sx, y: sy };
}
