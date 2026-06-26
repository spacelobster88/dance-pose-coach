/**
 * Synchronized playback of two <video> elements (reference + test).
 *
 * v0.1 alignment is by *playback progress*: both clips share play/pause and
 * the test clip is seeked to the reference's normalized progress whenever they
 * drift apart by more than a small threshold. This keeps frames loosely in
 * step without DTW. A per-frame callback drives detection + scoring.
 *
 * When the test source is a live webcam stream there is no seekable timeline,
 * so `liveTest` disables all test-seeking: the reference plays as the routine
 * and each animation frame simply reads the current camera frame.
 */

export interface DualPlayerCallbacks {
  /** Called once per detection tick while playing, after sync correction. */
  onFrame: (ref: HTMLVideoElement, test: HTMLVideoElement) => void;
  onPlay?: () => void;
  onPause?: () => void;
  onEnded?: () => void;
}

const DRIFT_THRESHOLD_SEC = 0.12;

/**
 * Cadence cap for the rAF fallback path (browsers without
 * `requestVideoFrameCallback`). Detection is the heavy work — two inferences +
 * scoring + draw — and the render loop fires at 60–120fps, far faster than a
 * dance clip's ~24–30fps decoded-frame rate, so running detection every render
 * frame just saturates the main thread (the jank + score-lag root cause). We
 * cap the fallback at ~24fps; the rVFC path instead infers once per *decoded*
 * frame, which is naturally throttled to the source's frame rate.
 */
const DETECT_FALLBACK_INTERVAL_MS = 1000 / 24;

/** A <video> that supports the (still un-typed in our lib) rVFC API. */
type RvfcVideo = HTMLVideoElement & {
  requestVideoFrameCallback(
    cb: (now: number, metadata: unknown) => void,
  ): number;
  cancelVideoFrameCallback(handle: number): void;
};

function asRvfc(v: HTMLVideoElement): RvfcVideo | null {
  return typeof (v as RvfcVideo).requestVideoFrameCallback === "function"
    ? (v as RvfcVideo)
    : null;
}

export class DualPlayer {
  readonly ref: HTMLVideoElement;
  readonly test: HTMLVideoElement;
  private rafId: number | null = null;
  private rvfcId: number | null = null;
  // The reference clip drives the loop; prefer per-decoded-frame callbacks when
  // available so detection tracks the actual decoded frames, not the render rate.
  private rvfc: RvfcVideo | null = null;
  private lastDetectMs = 0;
  private callbacks: DualPlayerCallbacks;
  private running = false;
  private liveTest = false;
  private warp: ((refTime: number) => number) | null = null;

  constructor(
    ref: HTMLVideoElement,
    test: HTMLVideoElement,
    callbacks: DualPlayerCallbacks,
  ) {
    this.ref = ref;
    this.test = test;
    this.callbacks = callbacks;
    this.rvfc = asRvfc(ref);

    // When the reference ends, stop the loop and surface the event.
    this.ref.addEventListener("ended", () => {
      this.pause();
      this.callbacks.onEnded?.();
    });
  }

  /** True once both videos have enough data to render a frame. */
  get bothReady(): boolean {
    return this.ref.readyState >= 2 && this.test.readyState >= 2;
  }

  /**
   * Switch the test source between a seekable file (false) and a live webcam
   * stream (true). In live mode the test clip is never seeked.
   */
  setLiveTest(live: boolean): void {
    this.liveTest = live;
  }

  get isLiveTest(): boolean {
    return this.liveTest;
  }

  /**
   * Install a DTW warp mapping reference time → test time. When null, the test
   * clip is aligned by linear playback progress (the v0.1 behavior).
   */
  setWarp(warp: ((refTime: number) => number) | null): void {
    this.warp = warp;
  }

  async play(): Promise<void> {
    if (!this.bothReady) return;
    this.running = true;
    this.lastDetectMs = 0; // detect on the very first tick
    // Drive the test clip from the reference's progress, so align before play.
    if (!this.liveTest) this.syncTestToRef(true);
    await Promise.all([this.ref.play(), this.test.play()]);
    this.callbacks.onPlay?.();
    this.scheduleTick();
  }

  pause(): void {
    this.running = false;
    this.ref.pause();
    this.test.pause();
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    if (this.rvfcId !== null && this.rvfc) {
      this.rvfc.cancelVideoFrameCallback(this.rvfcId);
      this.rvfcId = null;
    }
    this.callbacks.onPause?.();
  }

  get playing(): boolean {
    return this.running;
  }

  /** Seek both clips back to the start (by progress) and render one frame. */
  restart(): void {
    this.ref.currentTime = 0;
    // A live webcam stream has no seekable timeline — only reset the reference.
    if (!this.liveTest) this.test.currentTime = 0;
    this.renderOnce();
  }

  /**
   * Seek the reference to `refTime` (seconds), bring the test clip to its
   * matched position (respecting any DTW warp), and render one frame. Pauses
   * first so the seek is a clean scrub — used by the post-run report to jump
   * both videos to a segment for side-by-side review.
   */
  seek(refTime: number): void {
    this.pause();
    const rDur = this.ref.duration;
    const t = isFinite(rDur) && rDur > 0 ? Math.max(0, Math.min(rDur, refTime)) : Math.max(0, refTime);
    this.ref.currentTime = t;
    this.syncTestToRef(true);
    this.renderOnce();
  }

  /**
   * Map the reference's playback progress (0..1) onto the test clip's duration
   * and seek the test clip there if it has drifted. When `force` is true the
   * seek happens regardless of drift (used before play / for scrubbing).
   */
  private syncTestToRef(force = false): void {
    if (this.liveTest) return;
    const rDur = this.ref.duration;
    const tDur = this.test.duration;
    if (!isFinite(rDur) || !isFinite(tDur) || rDur <= 0 || tDur <= 0) return;
    // With a DTW warp, map ref time → matched test time; otherwise align by
    // raw playback progress.
    let target: number;
    if (this.warp) {
      target = Math.max(0, Math.min(tDur, this.warp(this.ref.currentTime)));
    } else {
      target = (this.ref.currentTime / rDur) * tDur;
    }
    if (force || Math.abs(this.test.currentTime - target) > DRIFT_THRESHOLD_SEC) {
      this.test.currentTime = target;
    }
  }

  /** Run the per-frame callback exactly once (e.g. after a manual seek). */
  renderOnce(): void {
    if (!this.bothReady) return;
    this.callbacks.onFrame(this.ref, this.test);
  }

  /**
   * Schedule the next detection tick. Prefers `requestVideoFrameCallback` so we
   * infer once per *decoded* reference frame (naturally throttled to the clip's
   * frame rate, decoupled from the 60–120fps render loop); otherwise falls back
   * to rAF with a fixed cadence cap.
   */
  private scheduleTick(): void {
    if (this.rvfc) {
      this.rvfcId = this.rvfc.requestVideoFrameCallback((now) =>
        this.tick(now),
      );
    } else {
      this.rafId = requestAnimationFrame((now) => this.tick(now));
    }
  }

  private tick = (nowMs: number): void => {
    if (!this.running) return;
    // Drift correction is cheap; keep the test clip aligned every tick.
    this.syncTestToRef(false);
    // On the rAF fallback, throttle the heavy detection to a fixed cadence. The
    // rVFC path already fires once per decoded frame, so it runs every tick;
    // app.ts's `frameBusy` guard drops any tick that lands while an inference is
    // still in flight, so detection never overlaps itself regardless of path.
    const due =
      this.rvfc !== null ||
      nowMs - this.lastDetectMs >= DETECT_FALLBACK_INTERVAL_MS;
    if (due) {
      this.lastDetectMs = nowMs;
      this.callbacks.onFrame(this.ref, this.test);
    }
    this.scheduleTick();
  };

  dispose(): void {
    this.pause();
  }
}

/**
 * Point a <video> at a user-selected File and resolve once metadata (and thus
 * dimensions/duration) is available.
 */
export function loadVideoFromFile(
  video: HTMLVideoElement,
  file: File,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error(`Failed to load video: ${file.name}`));
    };
    const cleanup = () => {
      video.removeEventListener("loadeddata", onLoaded);
      video.removeEventListener("error", onError);
    };
    video.addEventListener("loadeddata", onLoaded);
    video.addEventListener("error", onError);
    // Revoke any previous object URL to avoid leaks.
    const prev = video.dataset.objectUrl;
    if (prev) URL.revokeObjectURL(prev);
    video.dataset.objectUrl = url;
    video.src = url;
    video.load();
  });
}
