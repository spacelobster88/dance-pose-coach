// Composites one scored comparison frame into a single canvas for export:
// the reference pane and the "your attempt" pane side by side (video + skeleton
// overlays, worst-limb highlight) with a score banner underneath. This is what
// the in-browser recorder (recorder.ts) captures into a downloadable clip.

import type { Pose } from "../pose/keypoints";
import { strokePose, type PoseTransform } from "./skeleton";

export const PANE_W = 640;
export const PANE_H = 480;
export const BANNER_H = 116;
export const COMPOSITE_W = PANE_W * 2;
export const COMPOSITE_H = PANE_H + BANNER_H;

export interface CompositeFrame {
  refVideo: HTMLVideoElement;
  refPose: Pose | null;
  testVideo: HTMLVideoElement;
  testPose: Pose | null;
  testLabel: string;
  scoreNow: number | null;
  scoreAvg: number | null;
  lagMs: number | null;
  worstEdges?: Array<[number, number]>;
}

const BG = "#16161a";
const PANE_BG = "#0e0e12";
const INK = "#f5f4f1";
const INK_FAINT = "#9a9aa2";

/** Ensure the export canvas is at the composite resolution. */
export function sizeComposite(canvas: HTMLCanvasElement): void {
  if (canvas.width !== COMPOSITE_W) canvas.width = COMPOSITE_W;
  if (canvas.height !== COMPOSITE_H) canvas.height = COMPOSITE_H;
}

/** Contain-fit a source w×h into a pane, returning the draw rect + scale. */
function containFit(vw: number, vh: number, paneX: number) {
  const scale = Math.min(PANE_W / vw, PANE_H / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const dx = paneX + (PANE_W - dw) / 2;
  const dy = (PANE_H - dh) / 2;
  return { scale, dw, dh, dx, dy };
}

function drawPane(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  pose: Pose | null,
  paneX: number,
  title: string,
  worstEdges?: Array<[number, number]>,
): void {
  ctx.fillStyle = PANE_BG;
  ctx.fillRect(paneX, 0, PANE_W, PANE_H);

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (vw && vh && video.readyState >= 2) {
    const { scale, dw, dh, dx, dy } = containFit(vw, vh, paneX);
    ctx.drawImage(video, dx, dy, dw, dh);
    if (pose) {
      const t: PoseTransform = { sx: scale, sy: scale, dx, dy };
      strokePose(ctx, pose, {
        transform: t,
        pointRadius: Math.max(3, dw * 0.008),
        lineWidth: Math.max(2, dw * 0.006),
        highlightEdges: worstEdges,
      });
    }
  }

  // Title chip.
  ctx.font = "600 18px Inter, ui-sans-serif, system-ui, sans-serif";
  ctx.textBaseline = "top";
  const tw = ctx.measureText(title).width;
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  ctx.fillRect(paneX + 16, 16, tw + 24, 30);
  ctx.fillStyle = INK;
  ctx.fillText(title, paneX + 28, 22);
}

export function drawComposite(
  canvas: HTMLCanvasElement,
  frame: CompositeFrame,
): void {
  sizeComposite(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, COMPOSITE_W, COMPOSITE_H);

  drawPane(ctx, frame.refVideo, frame.refPose, 0, "Reference");
  drawPane(
    ctx,
    frame.testVideo,
    frame.testPose,
    PANE_W,
    frame.testLabel,
    frame.worstEdges,
  );

  // Divider between panes.
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(PANE_W, 0);
  ctx.lineTo(PANE_W, PANE_H);
  ctx.stroke();

  drawBanner(ctx, frame);
}

function drawBanner(
  ctx: CanvasRenderingContext2D,
  frame: CompositeFrame,
): void {
  const y0 = PANE_H;
  ctx.fillStyle = BG;
  ctx.fillRect(0, y0, COMPOSITE_W, BANNER_H);

  const now = frame.scoreNow;
  const pct = now === null ? 0 : Math.max(0, Math.min(100, now));
  const hue = (pct / 100) * 120;

  // Big live score.
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = INK_FAINT;
  ctx.font = "600 13px Inter, ui-sans-serif, system-ui, sans-serif";
  ctx.fillText("SIMILARITY · LIVE", 28, y0 + 30);
  ctx.fillStyle = `hsl(${hue}, 80%, 55%)`;
  ctx.font = "700 52px Inter, ui-sans-serif, system-ui, sans-serif";
  ctx.fillText(now === null ? "—" : now.toFixed(0), 26, y0 + 86);

  // Secondary stats: average and (optional) lag.
  ctx.fillStyle = INK_FAINT;
  ctx.font = "600 13px Inter, ui-sans-serif, system-ui, sans-serif";
  ctx.fillText("AVERAGE", 200, y0 + 30);
  ctx.fillStyle = INK;
  ctx.font = "600 26px Inter, ui-sans-serif, system-ui, sans-serif";
  ctx.fillText(
    frame.scoreAvg === null ? "—" : frame.scoreAvg.toFixed(1),
    200,
    y0 + 62,
  );

  if (frame.lagMs !== null) {
    ctx.fillStyle = INK_FAINT;
    ctx.font = "600 13px Inter, ui-sans-serif, system-ui, sans-serif";
    ctx.fillText("LAG", 320, y0 + 30);
    ctx.fillStyle = INK;
    ctx.font = "600 26px Inter, ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(`${frame.lagMs.toFixed(0)}ms`, 320, y0 + 62);
  }

  // Score bar across the right portion of the banner.
  const barX = 470;
  const barW = COMPOSITE_W - barX - 28;
  const barY = y0 + BANNER_H / 2 - 7;
  const barH = 14;
  ctx.fillStyle = "rgba(255,255,255,0.08)";
  roundRect(ctx, barX, barY, barW, barH, 7);
  ctx.fill();
  ctx.fillStyle = `hsl(${hue}, 80%, 50%)`;
  roundRect(ctx, barX, barY, (barW * pct) / 100, barH, 7);
  ctx.fill();

  // Wordmark.
  ctx.fillStyle = INK_FAINT;
  ctx.font = "600 12px Inter, ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("dance-pose-coach", COMPOSITE_W - 28, y0 + BANNER_H - 16);
  ctx.textAlign = "left";
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}
