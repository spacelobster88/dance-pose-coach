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

export class DualPlayer {
  readonly ref: HTMLVideoElement;
  readonly test: HTMLVideoElement;
  private rafId: number | null = null;
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
    this.syncTestToRef(false);
    this.callbacks.onFrame(this.ref, this.test);
    this.rafId = requestAnimationFrame(this.loop);
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
