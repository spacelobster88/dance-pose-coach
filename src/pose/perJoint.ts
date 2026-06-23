// Per-limb divergence: the single 0–100 similarity score says *how* different
// two poses are, not *where*. This breaks the difference down by limb so the
// dancer knows which body part to fix, and exposes the worst-diverging limb so
// the renderer can highlight it on the skeleton overlay.

import { KEYPOINT_INDEX, type KeypointName } from "./keypoints";
import type { NormalizedPose } from "./normalize";

const I = KEYPOINT_INDEX;

export interface LimbGroup {
  key: string;
  label: string;
  /** Keypoints whose ref↔test distance is averaged for this limb. */
  joints: KeypointName[];
  /** Bone edges (keypoint index pairs) to highlight when this limb is worst. */
  edges: Array<[number, number]>;
}

/** Body grouping used for the breakdown (face keypoints are excluded). */
export const LIMB_GROUPS: LimbGroup[] = [
  {
    key: "left_arm",
    label: "Left arm",
    joints: ["left_shoulder", "left_elbow", "left_wrist"],
    edges: [
      [I.left_shoulder, I.left_elbow],
      [I.left_elbow, I.left_wrist],
    ],
  },
  {
    key: "right_arm",
    label: "Right arm",
    joints: ["right_shoulder", "right_elbow", "right_wrist"],
    edges: [
      [I.right_shoulder, I.right_elbow],
      [I.right_elbow, I.right_wrist],
    ],
  },
  {
    key: "left_leg",
    label: "Left leg",
    joints: ["left_hip", "left_knee", "left_ankle"],
    edges: [
      [I.left_hip, I.left_knee],
      [I.left_knee, I.left_ankle],
    ],
  },
  {
    key: "right_leg",
    label: "Right leg",
    joints: ["right_hip", "right_knee", "right_ankle"],
    edges: [
      [I.right_hip, I.right_knee],
      [I.right_knee, I.right_ankle],
    ],
  },
  {
    key: "torso",
    label: "Torso",
    joints: ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    edges: [
      [I.left_shoulder, I.right_shoulder],
      [I.left_shoulder, I.left_hip],
      [I.right_shoulder, I.right_hip],
      [I.left_hip, I.right_hip],
    ],
  },
];

export interface LimbDivergence {
  key: string;
  label: string;
  /**
   * Mean keypoint distance in normalized (torso = 1) units, or null when the
   * limb's keypoints were not confidently detected in both poses.
   */
  distance: number | null;
  edges: Array<[number, number]>;
}

export interface PerJointResult {
  limbs: LimbDivergence[];
  worst: LimbDivergence | null;
}

function keypointDistance(
  ref: NormalizedPose,
  test: NormalizedPose,
  idx: number,
): number | null {
  const p = ref.points[idx];
  const q = test.points[idx];
  if (!p || !q || !p.valid || !q.valid) return null;
  return Math.hypot(p.x - q.x, p.y - q.y);
}

function pickWorst(limbs: LimbDivergence[]): LimbDivergence | null {
  let worst: LimbDivergence | null = null;
  for (const l of limbs) {
    if (l.distance === null) continue;
    if (!worst || (worst.distance ?? -1) < l.distance) worst = l;
  }
  return worst;
}

/** Per-limb divergence for a single frame. */
export function perJointDivergence(
  ref: NormalizedPose,
  test: NormalizedPose,
): PerJointResult {
  const limbs: LimbDivergence[] = LIMB_GROUPS.map((g) => {
    let sum = 0;
    let cnt = 0;
    for (const name of g.joints) {
      const d = keypointDistance(ref, test, KEYPOINT_INDEX[name]);
      if (d !== null) {
        sum += d;
        cnt++;
      }
    }
    return {
      key: g.key,
      label: g.label,
      distance: cnt ? sum / cnt : null,
      edges: g.edges,
    };
  });
  return { limbs, worst: pickWorst(limbs) };
}

/**
 * Exponential-moving-average accumulator that smooths per-limb divergence over
 * frames so the breakdown bars and the highlighted limb don't flicker.
 */
export class LimbDivergenceTracker {
  private ema = new Map<string, number>();
  private readonly alpha: number;

  constructor(alpha = 0.3) {
    this.alpha = alpha;
  }

  push(result: PerJointResult): void {
    for (const l of result.limbs) {
      if (l.distance === null) continue;
      const prev = this.ema.get(l.key);
      this.ema.set(
        l.key,
        prev === undefined ? l.distance : this.alpha * l.distance + (1 - this.alpha) * prev,
      );
    }
  }

  smoothed(): LimbDivergence[] {
    return LIMB_GROUPS.map((g) => ({
      key: g.key,
      label: g.label,
      distance: this.ema.has(g.key) ? (this.ema.get(g.key) as number) : null,
      edges: g.edges,
    }));
  }

  worst(): LimbDivergence | null {
    return pickWorst(this.smoothed());
  }

  reset(): void {
    this.ema.clear();
  }
}
