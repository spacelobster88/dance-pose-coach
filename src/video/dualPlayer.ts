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
  /** Called once per animation frame while playing, after sync correction. */
  onFrame: (ref: HTMLVideoElement, test: HTMLVideoElement) => void;
  onPlay?: () => void;
  onPause?: () => void;
  onEnded?: () => void;
}

const DRIFT_THRESHOLD_SEC = 0.12;

/**
 * Detection cadence cap. The rAF loop still runs every frame to keep the test
 * clip synced to the reference (smooth video), but the expensive `onFrame`
 * detection+scoring callback fires at most once per this interval (~24fps).
 * Decoupling detection from the ~60fps render rate is what relieves the
 * main-thread saturation behind the jank + score latency in issue #14.
 */
const DETECT_INTERVAL_MS = 1000 / 24;

export class DualPlayer {
  readonly ref: HTMLVideoElement;
  readonly test: HTMLVideoElement;
  private rafId: number | null = null;
  private callbacks: DualPlayerCallbacks;
  private running = false;
  private liveTest = false;
  private warp: ((refTime: number) => number) | null = null;
  // Timestamp (performance.now ms) of the last detection callback, used to
  // throttle detection to DETECT_INTERVAL_MS independently of the render rate.
  private lastDetectionMs = 0;

  constructor(
    ref: HTMLVideoElement,
    test: HTMLVideoElement,
    callbacks: DualPlayerCallbacks,
  ) {
    this.ref = ref;
    this.test = test;
    this.callbacks = callbacks;

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
    // Force the first loop iteration to run detection immediately (don't wait
    // up to one throttle interval before the first scored frame).
    this.lastDetectionMs = this.nowMs() - DETECT_INTERVAL_MS;
    // Drive the test clip from the reference's progress, so align before play.
    if (!this.liveTest) this.syncTestToRef(true);
    await Promise.all([this.ref.play(), this.test.play()]);
    this.callbacks.onPlay?.();
    this.loop();
  }

  pause(): void {
    this.running = false;
    this.ref.pause();
    this.test.pause();
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
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

  private loop = (): void => {
    if (!this.running) return;
    // Keep the test clip synced to the reference every render frame (smooth
    // video), but run the heavy detection callback at most ~24fps.
    this.syncTestToRef(false);
    const now = this.nowMs();
    if (now - this.lastDetectionMs >= DETECT_INTERVAL_MS) {
      this.lastDetectionMs = now;
      this.callbacks.onFrame(this.ref, this.test);
    }
    this.rafId = requestAnimationFrame(this.loop);
  };

  /** Monotonic clock in ms, with a Date.now fallback for non-DOM contexts. */
  private nowMs(): number {
    return typeof performance !== "undefined" && performance.now
      ? performance.now()
      : Date.now();
  }

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
