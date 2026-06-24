// In-browser recorder for the scored comparison clip.
//
// Captures a canvas's live stream (the composite drawn each frame by
// composite.ts) with MediaRecorder and resolves a downloadable Blob on stop.
// Everything stays client-side — no upload, no server.

/** Container the user can ask to export. */
export type ExportFormat = "mp4" | "webm";

// MP4 (H.264) — natively recordable in Safari and recent Chromium; not in Firefox.
const MP4_CANDIDATES = [
  "video/mp4;codecs=avc1.42E01E", // H.264 baseline, broadest playback
  "video/mp4;codecs=avc1",
  "video/mp4",
];
// WebM (VP9/VP8) — the universal fallback supported by every MediaRecorder.
const WEBM_CANDIDATES = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
];

function isTypeSupported(t: string): boolean {
  const MR = (window as { MediaRecorder?: typeof MediaRecorder }).MediaRecorder;
  return Boolean(MR?.isTypeSupported?.(t));
}

/** True if this browser can record MP4 directly (no transcode needed). */
export function mp4Supported(): boolean {
  return MP4_CANDIDATES.some(isTypeSupported);
}

/**
 * Pick the best supported mime type for the requested container. Prefers the
 * requested family, but falls back to the other so recording never fails just
 * because (e.g.) Firefox can't write MP4.
 */
function pickMimeType(format: ExportFormat): string {
  const order =
    format === "mp4"
      ? [...MP4_CANDIDATES, ...WEBM_CANDIDATES]
      : [...WEBM_CANDIDATES, ...MP4_CANDIDATES];
  for (const t of order) {
    if (isTypeSupported(t)) return t;
  }
  return format === "mp4" ? "video/mp4" : "video/webm";
}

export function recordingSupported(): boolean {
  return (
    typeof MediaRecorder !== "undefined" &&
    typeof HTMLCanvasElement.prototype.captureStream === "function"
  );
}

export class ComparisonRecorder {
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private mimeType = "video/webm";

  get active(): boolean {
    return this.recorder !== null && this.recorder.state !== "inactive";
  }

  /** The container actually used (may differ from the request on fallback). */
  get extension(): ExportFormat {
    return this.mimeType.startsWith("video/mp4") ? "mp4" : "webm";
  }

  /** Begin capturing `canvas` at `fps` in `format`. Throws if unsupported. */
  start(canvas: HTMLCanvasElement, fps = 30, format: ExportFormat = "webm"): void {
    if (!recordingSupported()) {
      throw new Error("Recording is not supported in this browser.");
    }
    if (this.active) return;
    this.mimeType = pickMimeType(format);
    this.chunks = [];
    const stream = canvas.captureStream(fps);
    this.recorder = new MediaRecorder(stream, {
      mimeType: this.mimeType,
      videoBitsPerSecond: 6_000_000,
    });
    this.recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) this.chunks.push(e.data);
    };
    this.recorder.start(200); // emit chunks periodically so nothing is lost
  }

  /** Stop capturing and resolve the recorded clip as a Blob. */
  stop(): Promise<Blob> {
    return new Promise((resolve, reject) => {
      const rec = this.recorder;
      if (!rec || rec.state === "inactive") {
        reject(new Error("Recorder is not running."));
        return;
      }
      rec.onstop = () => {
        const blob = new Blob(this.chunks, { type: this.mimeType });
        this.recorder = null;
        this.chunks = [];
        resolve(blob);
      };
      rec.stop();
    });
  }
}

/** Trigger a browser download of `blob` as `filename`. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoke after the click has a chance to start the download.
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}
