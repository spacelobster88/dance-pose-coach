import { SCORING_KEYPOINT_INDICES } from "./keypoints";
import type { NormalizedPose } from "./normalize";

export interface ScoreResult {
  /** Raw cosine similarity in [-1, 1]. */
  cosine: number;
  /** Display score in [0, 100]. */
  score: number;
  /** How many keypoints were shared (valid in both poses) for this comparison. */
  sharedKeypoints: number;
}

/**
 * Cosine similarity between two normalized poses, computed over the body
 * (non-face) keypoints that are valid in *both* poses.
 *
 * Returns null when there is too little overlap to be meaningful.
 */
export function poseSimilarity(
  ref: NormalizedPose,
  test: NormalizedPose,
  minSharedKeypoints = 4,
): ScoreResult | null {
  const refVec: number[] = [];
  const testVec: number[] = [];
  let shared = 0;

  for (const idx of SCORING_KEYPOINT_INDICES) {
    const r = ref.points[idx];
    const t = test.points[idx];
    if (!r || !t || !r.valid || !t.valid) continue;
    refVec.push(r.x, r.y);
    testVec.push(t.x, t.y);
    shared++;
  }

  if (shared < minSharedKeypoints) return null;

  let dot = 0;
  let nr = 0;
  let nt = 0;
  for (let i = 0; i < refVec.length; i++) {
    dot += refVec[i] * testVec[i];
    nr += refVec[i] * refVec[i];
    nt += testVec[i] * testVec[i];
  }
  const denom = Math.sqrt(nr) * Math.sqrt(nt);
  if (denom < 1e-9) return null;

  const cosine = Math.max(-1, Math.min(1, dot / denom));
  // Map [-1, 1] -> [0, 100]. In practice normalized body poses cluster well
  // above 0, so the useful band is roughly 60–100, but we keep the full range.
  const score = ((cosine + 1) / 2) * 100;

  return { cosine, score, sharedKeypoints: shared };
}

/**
 * Exponential moving average accumulator for smoothing the live score and
 * tracking a running mean for the session summary.
 */
export class ScoreTracker {
  private ema: number | null = null;
  private sum = 0;
  private count = 0;
  private readonly alpha: number;

  constructor(alpha = 0.3) {
    this.alpha = alpha;
  }

  push(score: number): void {
    this.ema = this.ema === null ? score : this.alpha * score + (1 - this.alpha) * this.ema;
    this.sum += score;
    this.count++;
  }

  get smoothed(): number | null {
    return this.ema;
  }

  get average(): number | null {
    return this.count ? this.sum / this.count : null;
  }

  get samples(): number {
    return this.count;
  }

  reset(): void {
    this.ema = null;
    this.sum = 0;
    this.count = 0;
  }
}
