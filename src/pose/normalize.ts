import {
  KEYPOINT_INDEX,
  type Pose,
} from "./keypoints";

/**
 * A pose normalized into a translation- and scale-invariant frame:
 * centered on the mid-hip and scaled by torso length. Keypoints below the
 * confidence threshold are marked invalid so scoring can skip them.
 */
export interface NormalizedPose {
  /** Same length/order as the source keypoints. */
  points: Array<{ x: number; y: number; valid: boolean }>;
  /**
   * Optional 3D channel, present only when the source `Pose` carried
   * `worldKeypoints` (BlazePose, eng-B1). Same index order as `points`. When
   * present, scoring aligns the two poses in this viewpoint-invariant 3D frame
   * via Procrustes (procrustes.ts) before comparing; when absent the existing
   * 2D bone-direction path runs unchanged.
   */
  points3d?: Array<{ x: number; y: number; z: number; valid: boolean }>;
}

export const DEFAULT_MIN_KEYPOINT_SCORE = 0.3;

function midpoint(
  a: { x: number; y: number },
  b: { x: number; y: number },
): { x: number; y: number } {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function dist(
  a: { x: number; y: number },
  b: { x: number; y: number },
): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Normalize a pose so similarity scoring is invariant to where the dancer
 * stands (translation) and how large they appear (scale).
 *
 * - Origin = midpoint of the two hips.
 * - Scale = torso length = distance between mid-shoulder and mid-hip.
 *   Falls back to hip width, then to a bounding-box diagonal, so a usable
 *   scale exists even when some torso keypoints are missing.
 *
 * Returns null only when there is no viable anchor/scale at all.
 */
export function normalizePose(
  pose: Pose,
  minScore: number = DEFAULT_MIN_KEYPOINT_SCORE,
): NormalizedPose | null {
  const kp = pose.keypoints;
  if (kp.length < 17) return null;

  const lHip = kp[KEYPOINT_INDEX.left_hip];
  const rHip = kp[KEYPOINT_INDEX.right_hip];
  const lSho = kp[KEYPOINT_INDEX.left_shoulder];
  const rSho = kp[KEYPOINT_INDEX.right_shoulder];

  const hipOk =
    (lHip.score ?? 0) >= minScore && (rHip.score ?? 0) >= minScore;
  const shoOk =
    (lSho.score ?? 0) >= minScore && (rSho.score ?? 0) >= minScore;

  // Anchor: prefer mid-hip; fall back to mid-shoulder if hips are missing.
  let origin: { x: number; y: number };
  if (hipOk) {
    origin = midpoint(lHip, rHip);
  } else if (shoOk) {
    origin = midpoint(lSho, rSho);
  } else {
    return null;
  }

  // Scale: torso length, then hip width, then bbox diagonal of valid points.
  let scale = 0;
  if (hipOk && shoOk) {
    scale = dist(midpoint(lSho, rSho), midpoint(lHip, rHip));
  }
  if (scale < 1e-3 && hipOk) {
    scale = dist(lHip, rHip);
  }
  if (scale < 1e-3 && shoOk) {
    scale = dist(lSho, rSho);
  }
  if (scale < 1e-3) {
    // Last resort: bounding-box diagonal over confident keypoints.
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (const k of kp) {
      if ((k.score ?? 0) < minScore) continue;
      minX = Math.min(minX, k.x);
      minY = Math.min(minY, k.y);
      maxX = Math.max(maxX, k.x);
      maxY = Math.max(maxY, k.y);
    }
    if (maxX > minX || maxY > minY) {
      scale = Math.hypot(maxX - minX, maxY - minY);
    }
  }
  if (scale < 1e-3) return null;

  const points = kp.map((k) => {
    const valid = (k.score ?? 0) >= minScore;
    return {
      x: (k.x - origin.x) / scale,
      y: (k.y - origin.y) / scale,
      valid,
    };
  });

  // Attach the 3D channel when 3D world landmarks are available (BlazePose).
  // World landmarks are metric with origin at the hip center; we only translate
  // them to their own valid centroid and divide by the same torso-ish scale so
  // the magnitudes are comparable. Procrustes (similarity.ts) then removes the
  // remaining viewpoint rotation before comparison. 2D-only poses skip this.
  const world = pose.worldKeypoints;
  let points3d:
    | Array<{ x: number; y: number; z: number; valid: boolean }>
    | undefined;
  if (world && world.length >= kp.length) {
    const validFlags = kp.map((k) => (k.score ?? 0) >= minScore);
    let cx = 0,
      cy = 0,
      cz = 0,
      cn = 0;
    for (let i = 0; i < kp.length; i++) {
      if (!validFlags[i]) continue;
      const w = world[i];
      cx += w.x;
      cy += w.y;
      cz += w.z ?? 0;
      cn++;
    }
    if (cn > 0) {
      cx /= cn;
      cy /= cn;
      cz /= cn;
      // Scale by RMS radius of valid world points so units match roughly.
      let s = 0;
      for (let i = 0; i < kp.length; i++) {
        if (!validFlags[i]) continue;
        const w = world[i];
        const dx = w.x - cx;
        const dy = w.y - cy;
        const dz = (w.z ?? 0) - cz;
        s += dx * dx + dy * dy + dz * dz;
      }
      s = Math.sqrt(s / cn);
      if (s < 1e-9) s = 1;
      points3d = kp.map((_, i) => {
        const w = world[i];
        return {
          x: (w.x - cx) / s,
          y: (w.y - cy) / s,
          z: ((w.z ?? 0) - cz) / s,
          valid: validFlags[i],
        };
      });
    }
  }

  return points3d ? { points, points3d } : { points };
}
