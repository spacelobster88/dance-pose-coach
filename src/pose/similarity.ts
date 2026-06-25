import { SCORING_KEYPOINT_INDICES } from "./keypoints";
import { boneDirComparison, BONES } from "./boneAngles";
import type { NormalizedPose } from "./normalize";
import { procrustesAlign, type Point3 } from "./procrustes";

export interface ScoreResult {
  /**
   * Mean bone-direction cosine in [-1, 1] (mean cosθ across bones valid in both
   * poses). Kept for backward compatibility with anything reading `.cosine`.
   */
  cosine: number;
  /** Display score in [0, 100]. */
  score: number;
  /** How many keypoints were shared (valid in both poses) for this comparison. */
  sharedKeypoints: number;
}

/**
 * Strictness coefficient `k` for the exponential-decay scoring curve
 * `score = 100 * exp(-k * meanError)`, where `meanError` is the mean
 * bone-direction error (1 - cosθ) in [0, 2].
 *
 * Chosen so that an identical pose (meanError ≈ 0) scores ≈100, while a
 * clearly-wrong pose such as arms-up vs arms-down (a couple of bones flipped,
 * pushing meanError up) lands well below ~40. Exposed to the UI via a slider
 * (uiux-1 / eng-A2) through setStrictness / getStrictness.
 */
const DEFAULT_STRICTNESS = 6;
let strictness = DEFAULT_STRICTNESS;

/** Set the global strictness coefficient `k` (higher = harsher scoring). */
export function setStrictness(k: number): void {
  if (Number.isFinite(k) && k > 0) strictness = k;
}

/** Current strictness coefficient `k`. */
export function getStrictness(): number {
  return strictness;
}

const MIN_BONE_LEN_3D = 1e-6;

/** A 3D bone unit-direction with validity (3D analogue of boneAngles' BoneVector). */
interface BoneVector3 {
  x: number;
  y: number;
  z: number;
  valid: boolean;
}

/** Per-bone 3D unit directions over the shared BONES list. */
function boneVectors3(points: Point3[]): BoneVector3[] {
  return BONES.map((b) => {
    const a = points[b.from];
    const c = points[b.to];
    if (!a || !c || !a.valid || !c.valid) {
      return { x: 0, y: 0, z: 0, valid: false };
    }
    const dx = c.x - a.x;
    const dy = c.y - a.y;
    const dz = c.z - a.z;
    const len = Math.hypot(dx, dy, dz);
    if (len < MIN_BONE_LEN_3D) return { x: 0, y: 0, z: 0, valid: false };
    return { x: dx / len, y: dy / len, z: dz / len, valid: true };
  });
}

/**
 * Mean 3D bone-direction error between a reference pose and a test pose that has
 * ALREADY been Procrustes-aligned into the reference frame. Mirrors
 * boneDirComparison's (1 - cosθ) accumulation but in 3D. Returns null when no
 * bone is shared.
 */
function boneDirComparison3(
  ref: Point3[],
  alignedTest: Point3[],
): { meanError: number; meanCosine: number; sharedBones: number } | null {
  const rv = boneVectors3(ref);
  const tv = boneVectors3(alignedTest);
  let sumOneMinusCos = 0;
  let sumCos = 0;
  let shared = 0;
  for (let i = 0; i < rv.length; i++) {
    const r = rv[i];
    const t = tv[i];
    if (!r.valid || !t.valid) continue;
    let cos = r.x * t.x + r.y * t.y + r.z * t.z;
    if (cos > 1) cos = 1;
    else if (cos < -1) cos = -1;
    sumOneMinusCos += 1 - cos;
    sumCos += cos;
    shared++;
  }
  if (shared === 0) return null;
  return {
    meanError: sumOneMinusCos / shared,
    meanCosine: sumCos / shared,
    sharedBones: shared,
  };
}

/**
 * Similarity between two normalized poses, scored from a viewpoint-tolerant
 * bone-direction error passed through an exponential-decay strictness curve.
 *
 * Unlike cosine of raw normalized coordinates (which has a high floor because
 * pose coordinates are intrinsically correlated), bone *directions* cancel
 * translation and scale and discriminate grossly different poses strongly.
 *
 * - `meanError` = mean (1 - cosθ) over bones valid in both poses (boneAngles.ts)
 * - `score = 100 * exp(-k * meanError)` with configurable `k` (setStrictness)
 *
 * Returns null when there is too little overlap to be meaningful.
 *
 * @param k optional per-call strictness override; falls back to the module value.
 */
export function poseSimilarity(
  ref: NormalizedPose,
  test: NormalizedPose,
  minSharedKeypoints = 4,
  k = strictness,
): ScoreResult | null {
  // Count keypoints valid in both poses (backward-compatible overlap gate).
  let sharedKeypoints = 0;
  for (const idx of SCORING_KEYPOINT_INDICES) {
    const r = ref.points[idx];
    const t = test.points[idx];
    if (r && t && r.valid && t.valid) sharedKeypoints++;
  }
  if (sharedKeypoints < minSharedKeypoints) return null;

  // 3D path: when BOTH poses carry world landmarks, remove the viewpoint
  // difference by Procrustes-aligning test onto ref in 3D, then compare bone
  // directions in that canonical frame. Two camera angles of the same pose then
  // score ~100. Falls through to the 2D path if alignment isn't viable.
  if (ref.points3d && test.points3d) {
    const aligned = procrustesAlign(ref.points3d, test.points3d, false);
    if (aligned) {
      const cmp3 = boneDirComparison3(ref.points3d, aligned.points);
      if (cmp3) {
        const score3 = 100 * Math.exp(-k * cmp3.meanError);
        return {
          cosine: cmp3.meanCosine,
          score: Math.max(0, Math.min(100, score3)),
          sharedKeypoints,
        };
      }
    }
  }

  const cmp = boneDirComparison(ref, test);
  // No comparable bone (e.g. only isolated keypoints shared) -> not meaningful.
  if (!cmp) return null;

  const score = 100 * Math.exp(-k * cmp.meanError);

  return {
    cosine: cmp.meanCosine,
    score: Math.max(0, Math.min(100, score)),
    sharedKeypoints,
  };
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
