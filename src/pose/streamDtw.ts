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
import type { NormalizedPose } from "./normalize";

export interface StreamMatch {
  /** Best similarity score in [0, 100] within the lag window. */
  score: number;
  /** The reference pose that best matched the live frame. */
  refPose: NormalizedPose;
  /** How far behind the reference the live frame is, in ms (>= 0). */
  lagMs: number;
}

interface Sample {
  t: number; // reference timeline time, ms
  pose: NormalizedPose;
}

export class StreamingAligner {
  private buf: Sample[] = [];
  private lastMatchT = -Infinity;

  /**
   * @param maxLagMs   how far behind the reference a live frame may be matched
   * @param backSlackMs small backward tolerance so brief jitter doesn't lock the
   *                    matched time from correcting itself
   */
  constructor(
    private readonly maxLagMs = 700,
    private readonly backSlackMs = 120,
  ) {}

  /** Record the current reference pose at reference-timeline time `t` (ms). */
  pushRef(pose: NormalizedPose, t: number): void {
    // Keep the buffer time-ordered; ignore out-of-order timestamps (e.g. after a
    // restart the caller should reset() rather than push backwards).
    if (this.buf.length && t < this.buf[this.buf.length - 1].t) return;
    this.buf.push({ t, pose });
    const cutoff = t - this.maxLagMs;
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
    // Don't match earlier than (last matched time - slack), and never older than
    // the lag window — this enforces monotonic forward progress.
    const lowerT = Math.max(this.lastMatchT - this.backSlackMs, latestT - this.maxLagMs);

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
    return {
      score: bestScore,
      refPose: best.pose,
      lagMs: Math.max(0, latestT - best.t),
    };
  }

  reset(): void {
    this.buf = [];
    this.lastMatchT = -Infinity;
  }
}
