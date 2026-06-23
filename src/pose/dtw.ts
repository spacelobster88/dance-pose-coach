// Dynamic Time Warping alignment between two pose sequences.
//
// v0.1 compared frames at equal playback progress, so a slower or faster
// attempt scored badly even when the movement was correct. DTW instead finds
// the lowest-cost monotonic mapping between the reference and test pose
// sequences, so each reference frame is matched to the test frame whose pose is
// closest — making the score robust to tempo differences.

import { SCORING_KEYPOINT_INDICES } from "./keypoints";
import type { NormalizedPose } from "./normalize";

/** Minimum shared keypoints for a meaningful distance; below this we penalize. */
const MIN_SHARED = 4;
/** Distance used when two poses can't be compared (max mismatch). */
const MAX_DISTANCE = 1;

/**
 * Distance in [0, 1] between two normalized poses: (1 - cosine) / 2 over the
 * body (non-face) keypoints valid in both. Missing/invalid poses return the
 * maximum distance so DTW avoids matching through them when alternatives exist.
 */
export function poseDistance(
  a: NormalizedPose | null,
  b: NormalizedPose | null,
): number {
  if (!a || !b) return MAX_DISTANCE;
  let dot = 0;
  let na = 0;
  let nb = 0;
  let shared = 0;
  for (const idx of SCORING_KEYPOINT_INDICES) {
    const p = a.points[idx];
    const q = b.points[idx];
    if (!p || !q || !p.valid || !q.valid) continue;
    dot += p.x * q.x + p.y * q.y;
    na += p.x * p.x + p.y * p.y;
    nb += q.x * q.x + q.y * q.y;
    shared++;
  }
  if (shared < MIN_SHARED) return MAX_DISTANCE;
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  if (denom < 1e-9) return MAX_DISTANCE;
  const cosine = Math.max(-1, Math.min(1, dot / denom));
  return (1 - cosine) / 2;
}

export interface DtwResult {
  /** For each reference sample index, the matched test sample index. */
  refToTest: number[];
  /** Mean per-step distance along the warp path (lower = better overall match). */
  cost: number;
}

/**
 * Align two pose sequences with banded DTW (Sakoe-Chiba) and return, for every
 * reference sample, the matched test sample index. The band keeps cost and
 * memory linear-ish for long clips while still allowing realistic tempo drift.
 */
export function dtwAlign(
  refSeq: Array<NormalizedPose | null>,
  testSeq: Array<NormalizedPose | null>,
  windowFrac = 0.25,
): DtwResult {
  const n = refSeq.length;
  const m = testSeq.length;
  if (n === 0 || m === 0) return { refToTest: [], cost: Infinity };

  const w =
    Math.max(Math.abs(n - m), Math.ceil(Math.max(n, m) * windowFrac)) + 1;
  const INF = Infinity;
  const D: number[][] = Array.from({ length: n }, () =>
    new Array<number>(m).fill(INF),
  );

  const center = (i: number) => Math.floor((i * m) / n);
  for (let i = 0; i < n; i++) {
    const jStart = Math.max(0, center(i) - w);
    const jEnd = Math.min(m - 1, center(i) + w);
    for (let j = jStart; j <= jEnd; j++) {
      const cost = poseDistance(refSeq[i], testSeq[j]);
      let best: number;
      if (i === 0 && j === 0) {
        best = 0;
      } else {
        best = INF;
        if (i > 0) best = Math.min(best, D[i - 1][j]);
        if (j > 0) best = Math.min(best, D[i][j - 1]);
        if (i > 0 && j > 0) best = Math.min(best, D[i - 1][j - 1]);
      }
      D[i][j] = cost + best;
    }
  }

  // If the band was too tight to reach the corner, fall back to a linear map.
  if (!isFinite(D[n - 1][m - 1])) {
    const refToTest = new Array<number>(n);
    for (let k = 0; k < n; k++) {
      refToTest[k] = Math.round((k * (m - 1)) / Math.max(1, n - 1));
    }
    return { refToTest, cost: Infinity };
  }

  // Backtrack the optimal path, always stepping toward the finite minimum.
  const pairs: Array<[number, number]> = [];
  let i = n - 1;
  let j = m - 1;
  while (i > 0 || j > 0) {
    pairs.push([i, j]);
    if (i === 0) {
      j--;
      continue;
    }
    if (j === 0) {
      i--;
      continue;
    }
    const diag = D[i - 1][j - 1];
    const up = D[i - 1][j];
    const left = D[i][j - 1];
    if (diag <= up && diag <= left) {
      i--;
      j--;
    } else if (up <= left) {
      i--;
    } else {
      j--;
    }
  }
  pairs.push([0, 0]);

  // Average the matched test indices per reference index, filling any gaps with
  // the last seen value so the map is defined for every reference frame.
  const sum = new Array<number>(n).fill(0);
  const cnt = new Array<number>(n).fill(0);
  for (const [pi, pj] of pairs) {
    sum[pi] += pj;
    cnt[pi] += 1;
  }
  const refToTest = new Array<number>(n);
  let last = 0;
  for (let k = 0; k < n; k++) {
    if (cnt[k] > 0) {
      refToTest[k] = Math.round(sum[k] / cnt[k]);
      last = refToTest[k];
    } else {
      refToTest[k] = last;
    }
  }

  return { refToTest, cost: D[n - 1][m - 1] / pairs.length };
}
