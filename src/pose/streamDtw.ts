// Streaming / online time-warp for the live webcam follow mode.
//
// Offline DTW (dtw.ts) needs both sequences in full, so it can't run against a
// live camera. But a person dancing along to a reference is almost always a
// little *behind* it. Comparing the instantaneous reference frame to the camera
// frame then penalizes correct-but-lagging movement.
//
// StreamingAligner keeps a short sliding window of recent reference poses and,
// for each incoming live pose, picks the reference pose in that window it best
// matches — under a monotonicity constraint so the matched reference time only
// moves forward (it can't snap back to an earlier moment). The result is a
// lag-compensated score plus the measured lag, computed online in O(window).

import { poseSimilarity } from "./similarity";
import { getTransportOffsetMs, reactionLag } from "./syncCalib";
import type { NormalizedPose } from "./normalize";

export interface StreamMatch {
  /** Best similarity score in [0, 100] within the lag window. */
  score: number;
  /** The reference pose that best matched the live frame. */
  refPose: NormalizedPose;
  /**
   * The dancer's *reaction* lag in ms (>= 0): how late the human is, with the
   * measured machine transport delay removed. Equal to `totalLagMs` when no
   * calibration has run (transport offset 0).
   */
  lagMs: number;
  /**
   * Raw total observed lag in ms (>= 0): transport + reaction conflated, i.e.
   * `latestRefT - matchedRefT`. Optional; present for diagnostics / UI that
   * wants the uncompensated figure.
   */
  totalLagMs?: number;
  /**
   * The transport (machine) delay in ms subtracted from `totalLagMs` to obtain
   * `lagMs`, as read from the calibration seam at match time. 0 when
   * uncalibrated. Optional.
   */
  transportMs?: number;
}

/**
 * How much headroom (ms) to add on top of the measured transport delay when
 * widening the lag window, so a match landing right at the transport distance
 * still has room for genuine human reaction lag beyond it.
 */
const WINDOW_HEADROOM_MS = 250;

/**
 * Hard upper bound (ms) on the adaptively-widened lag window. Keeps the buffer
 * and per-frame scan bounded even with an implausibly large transport estimate.
 */
const MAX_EFFECTIVE_LAG_MS = 4000;

interface Sample {
  t: number; // reference timeline time, ms
  pose: NormalizedPose;
}

export class StreamingAligner {
  private buf: Sample[] = [];
  private lastMatchT = -Infinity;

  /**
   * @param baseMaxLagMs how far behind the reference a live frame may be matched
   *                     *before* accounting for transport. The effective window
   *                     is widened adaptively to cover the measured transport
   *                     delay (see {@link effectiveMaxLagMs}).
   * @param backSlackMs  small backward tolerance so brief jitter doesn't lock the
   *                     matched time from correcting itself
   * @param offsetSource source of the measured transport (machine) delay in ms.
   *                     Defaults to the syncCalib seam ({@link getTransportOffsetMs});
   *                     injectable so the aligner is unit-testable without globals.
   *                     Read live on every match so post-calibration updates take
   *                     effect immediately.
   */
  constructor(
    private readonly baseMaxLagMs = 700,
    private readonly backSlackMs = 120,
    private readonly offsetSource: () => number = getTransportOffsetMs,
  ) {}

  /** Current measured transport delay (ms), clamped to a finite >= 0 value. */
  private transportMs(): number {
    const v = this.offsetSource();
    return Number.isFinite(v) && v > 0 ? v : 0;
  }

  /**
   * Adaptive lag window (ms): the base window widened to cover the measured
   * transport delay plus a reaction headroom, capped at a sane upper bound. With
   * no calibration (transport 0) this collapses to `baseMaxLagMs`, preserving the
   * original behavior exactly.
   */
  private effectiveMaxLagMs(): number {
    const transport = this.transportMs();
    // No calibration -> exactly the base window (backward compatible). Once a
    // transport delay is measured, widen by transport + a reaction headroom.
    const widened =
      transport > 0 ? this.baseMaxLagMs + transport + WINDOW_HEADROOM_MS : this.baseMaxLagMs;
    return Math.min(MAX_EFFECTIVE_LAG_MS, Math.max(this.baseMaxLagMs, widened));
  }

  /** Record the current reference pose at reference-timeline time `t` (ms). */
  pushRef(pose: NormalizedPose, t: number): void {
    // Keep the buffer time-ordered; ignore out-of-order timestamps (e.g. after a
    // restart the caller should reset() rather than push backwards).
    if (this.buf.length && t < this.buf[this.buf.length - 1].t) return;
    this.buf.push({ t, pose });
    const cutoff = t - this.effectiveMaxLagMs();
    let drop = 0;
    while (drop < this.buf.length && this.buf[drop].t < cutoff) drop++;
    if (drop) this.buf.splice(0, drop);
  }

  /**
   * Match a live pose against the buffered reference window. Returns the best
   * match (and its lag), or null if there's no reference history yet.
   */
  match(live: NormalizedPose): StreamMatch | null {
    if (!this.buf.length) return null;
    const latestT = this.buf[this.buf.length - 1].t;
    const maxLag = this.effectiveMaxLagMs();
    // Don't match earlier than (last matched time - slack), and never older than
    // the (adaptively widened) lag window — this enforces monotonic forward
    // progress while still admitting matches as far back as real transport reaches.
    const lowerT = Math.max(this.lastMatchT - this.backSlackMs, latestT - maxLag);

    let best: Sample | null = null;
    let bestScore = -Infinity;
    for (const s of this.buf) {
      if (s.t < lowerT) continue;
      const r = poseSimilarity(s.pose, live);
      if (r && r.score > bestScore) {
        bestScore = r.score;
        best = s;
      }
    }
    if (!best) return null;

    this.lastMatchT = best.t;
    // Total observed lag conflates machine transport + human reaction; subtract
    // the measured transport delay to report the dancer's reaction lag alone.
    const totalLagMs = Math.max(0, latestT - best.t);
    const transportMs = this.transportMs();
    return {
      score: bestScore,
      refPose: best.pose,
      lagMs: reactionLag(totalLagMs, transportMs),
      totalLagMs,
      transportMs,
    };
  }

  reset(): void {
    this.buf = [];
    this.lastMatchT = -Infinity;
  }
}
