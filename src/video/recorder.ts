// In-browser recorder for the scored comparison clip.
//
// Captures a canvas's live stream (the composite drawn each frame by
// composite.ts) with MediaRecorder and resolves a downloadable Blob on stop.
// Everything stays client-side — no upload, no server.

/** Pick the best webm codec the browser actually supports. */
function pickMimeType(): string {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  const supported = (window as { MediaRecorder?: typeof MediaRecorder })
    .MediaRecorder;
  for (const t of candidates) {
    if (supported?.isTypeSupported?.(t)) return t;
  }
  return "video/webm";
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

  /** Begin capturing `canvas` at `fps`. Throws if recording is unsupported. */
  start(canvas: HTMLCanvasElement, fps = 30): void {
    if (!recordingSupported()) {
      throw new Error("Recording is not supported in this browser.");
    }
    if (this.active) return;
    this.mimeType = pickMimeType();
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
