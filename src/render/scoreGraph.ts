/**
 * A lightweight, DPI-aware sparkline of the similarity score over the course of
 * a routine. Keeps a fixed-size ring buffer of the most recent samples and
 * redraws on every push, so it stays cheap to call once per analysed frame.
 *
 * Two traces are drawn: the raw per-frame score (faint) and the EMA-smoothed
 * score (bold, hue-coded to match the scoreboard fill).
 */

export interface ScoreGraphOptions {
  /** How many samples to retain on screen. Older points scroll off the left. */
  capacity?: number;
}

const PAD = { top: 8, right: 8, bottom: 8, left: 8 };
// Only the meaningful band is shown — normalized body poses rarely score below
// 50, so stretching 50–100 across the height makes differences legible.
const Y_MIN = 50;
const Y_MAX = 100;
const GRIDLINES = [50, 75, 100];

export class ScoreGraph {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly capacity: number;
  private readonly raw: number[] = [];
  private readonly smooth: number[] = [];

  constructor(canvas: HTMLCanvasElement, opts: ScoreGraphOptions = {}) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("ScoreGraph: 2d context unavailable");
    this.canvas = canvas;
    this.ctx = ctx;
    this.capacity = Math.max(16, opts.capacity ?? 240);
    this.draw();
  }

  /** Append one frame's scores (both in 0–100) and redraw. */
  push(raw: number, smooth: number): void {
    this.raw.push(raw);
    this.smooth.push(smooth);
    if (this.raw.length > this.capacity) this.raw.shift();
    if (this.smooth.length > this.capacity) this.smooth.shift();
    this.draw();
  }

  reset(): void {
    this.raw.length = 0;
    this.smooth.length = 0;
    this.draw();
  }

  /** Resize the backing buffer to the element's CSS size at device pixel ratio. */
  private syncSize(): { w: number; h: number } {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const bw = w * dpr;
    const bh = h * dpr;
    if (this.canvas.width !== bw || this.canvas.height !== bh) {
      this.canvas.width = bw;
      this.canvas.height = bh;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
  }

  private yFor(score: number, h: number): number {
    const t = (score - Y_MIN) / (Y_MAX - Y_MIN);
    const clamped = Math.max(0, Math.min(1, t));
    const plotH = h - PAD.top - PAD.bottom;
    return PAD.top + (1 - clamped) * plotH;
  }

  private xFor(i: number, count: number, w: number): number {
    const plotW = w - PAD.left - PAD.right;
    // Pin the series to the right edge so the latest sample is always visible;
    // a partially-filled buffer grows from the right.
    const span = this.capacity - 1;
    const offset = this.capacity - count;
    return PAD.left + ((offset + i) / span) * plotW;
  }

  private draw(): void {
    const { w, h } = this.syncSize();
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    // Gridlines + labels.
    ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (const g of GRIDLINES) {
      const y = this.yFor(g, h);
      ctx.strokeStyle = "rgba(20, 20, 26, 0.07)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(w - PAD.right, y);
      ctx.stroke();
      ctx.fillStyle = "rgba(20, 20, 26, 0.32)";
      ctx.fillText(String(g), PAD.left + 2, y - 6);
    }

    if (this.smooth.length < 2) {
      // Nothing to draw yet — leave the empty grid as a placeholder.
      return;
    }

    // Raw trace (faint).
    this.trace(this.raw, w, h, "rgba(20, 20, 26, 0.16)", 1);

    // Smoothed trace (bold, hue-coded by the latest value).
    const last = this.smooth[this.smooth.length - 1];
    const pct = Math.max(0, Math.min(100, last));
    const hue = (pct / 100) * 120;
    this.trace(this.smooth, w, h, `hsl(${hue}, 80%, 42%)`, 2);

    // Marker dot on the latest smoothed sample.
    const lastX = this.xFor(this.smooth.length - 1, this.smooth.length, w);
    const lastY = this.yFor(last, h);
    ctx.fillStyle = `hsl(${hue}, 80%, 42%)`;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  private trace(
    data: number[],
    w: number,
    h: number,
    stroke: string,
    width: number,
  ): void {
    const ctx = this.ctx;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = this.xFor(i, data.length, w);
      const y = this.yFor(data[i], h);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}
