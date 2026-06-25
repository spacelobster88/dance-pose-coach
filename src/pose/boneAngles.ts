// Viewpoint-tolerant pose feature: bone direction unit vectors.
//
// Cosine similarity of normalized *absolute* keypoint coordinates has a high
// floor (~75) because human pose coordinates are intrinsically correlated, so
// grossly different poses still score high. Bone *directions* are a far better
// feature: a bone is the unit vector from its parent joint to its child joint,
// which cancels global translation and scale outright and is much less sensitive
// to viewpoint than raw coordinates. Comparing poses by the angles between
// matching bones gives a strict, discriminative error signal.

import { KEYPOINT_INDEX } from "./keypoints";
import type { NormalizedPose } from "./normalize";

const I = KEYPOINT_INDEX;

/** A directed bone: unit vector points from `from` joint toward `to` joint. */
export interface BoneDef {
  key: string;
  from: number;
  to: number;
}

/**
 * Fixed bone list consistent with the limb edges in perJoint.ts / keypoints.ts:
 * both arms (upper + fore), both legs (thigh + shin), and the torso quad
 * (shoulders, both sides, hips). Face keypoints are excluded, matching the
 * SCORING_KEYPOINTS rationale.
 */
export const BONES: BoneDef[] = [
  // Left arm
  { key: "l_upper_arm", from: I.left_shoulder, to: I.left_elbow },
  { key: "l_forearm", from: I.left_elbow, to: I.left_wrist },
  // Right arm
  { key: "r_upper_arm", from: I.right_shoulder, to: I.right_elbow },
  { key: "r_forearm", from: I.right_elbow, to: I.right_wrist },
  // Left leg
  { key: "l_thigh", from: I.left_hip, to: I.left_knee },
  { key: "l_shin", from: I.left_knee, to: I.left_ankle },
  // Right leg
  { key: "r_thigh", from: I.right_hip, to: I.right_knee },
  { key: "r_shin", from: I.right_knee, to: I.right_ankle },
  // Torso
  { key: "shoulders", from: I.left_shoulder, to: I.right_shoulder },
  { key: "hips", from: I.left_hip, to: I.right_hip },
  { key: "l_side", from: I.left_shoulder, to: I.left_hip },
  { key: "r_side", from: I.right_shoulder, to: I.right_hip },
];

/** One bone's direction as a unit vector, or invalid when undetermined. */
export interface BoneVector {
  key: string;
  /** Unit-vector x; 0 when invalid. */
  x: number;
  /** Unit-vector y; 0 when invalid. */
  y: number;
  /** True when both endpoints were valid and the bone had non-zero length. */
  valid: boolean;
}

const MIN_BONE_LEN = 1e-6;

/**
 * Compute per-bone unit direction vectors for a normalized pose. The result is
 * a fixed-length vector aligned with BONES, each entry carrying a validity flag
 * (false when either endpoint keypoint is missing or the bone is degenerate).
 */
export function boneVectors(pose: NormalizedPose): BoneVector[] {
  return BONES.map((b) => {
    const a = pose.points[b.from];
    const c = pose.points[b.to];
    if (!a || !c || !a.valid || !c.valid) {
      return { key: b.key, x: 0, y: 0, valid: false };
    }
    const dx = c.x - a.x;
    const dy = c.y - a.y;
    const len = Math.hypot(dx, dy);
    if (len < MIN_BONE_LEN) {
      return { key: b.key, x: 0, y: 0, valid: false };
    }
    return { key: b.key, x: dx / len, y: dy / len, valid: true };
  });
}

/** Per-bone comparison stats between two poses over their shared valid bones. */
export interface BoneDirComparison {
  /**
   * Mean of (1 - cosθ) across bones valid in both poses, in [0, 2]. 0 means
   * every shared bone points in exactly the same direction (identical pose);
   * larger means more divergent. This is the strict, viewpoint-robust error.
   */
  meanError: number;
  /** Mean cosθ across shared bones, in [-1, 1]; for backward-compat reporting. */
  meanCosine: number;
  /** How many bones were valid (comparable) in both poses. */
  sharedBones: number;
}

/**
 * Mean bone-direction error between two poses. For each bone valid in *both*
 * poses we take the cosine of the angle between the two unit vectors and
 * accumulate (1 - cosθ). The mean is 0 for identical poses and grows toward 2
 * as bones point in opposite directions.
 *
 * Returns null when no bone is shared (caller decides how to handle).
 */
export function boneDirComparison(
  ref: NormalizedPose,
  test: NormalizedPose,
): BoneDirComparison | null {
  const rv = boneVectors(ref);
  const tv = boneVectors(test);
  let sumOneMinusCos = 0;
  let sumCos = 0;
  let shared = 0;
  for (let i = 0; i < rv.length; i++) {
    const r = rv[i];
    const t = tv[i];
    if (!r.valid || !t.valid) continue;
    // Both are unit vectors, so the dot product is already cosθ in [-1, 1].
    let cos = r.x * t.x + r.y * t.y;
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
