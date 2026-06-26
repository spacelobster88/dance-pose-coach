// Run-report aggregation: turns the per-frame scoring + per-limb divergence that
// the app already computes into the compact, derived CoachingInput consumed by
// the insights providers.
//
// #9's rule-based report lives conceptually here: we segment the routine over
// time, express each limb's error in *degrees* of bone-direction divergence
// (more intuitive for a coach than torso-normalized distance) and a *signed*
// correction direction, and surface the top correction opportunities.

import { boneVectors, BONES } from "../pose/boneAngles";
import { LIMB_GROUPS } from "../pose/perJoint";
import { KEYPOINT_INDEX } from "../pose/keypoints";
import type { NormalizedPose } from "../pose/normalize";
import type {
  CoachingInput,
  LimbCorrection,
  ReportSegment,
  ScorePoint,
} from "./types";

const RAD2DEG = 180 / Math.PI;

/** Map each limb group to the indices of its bones within BONES. */
const LIMB_BONES: Record<string, number[]> = {
  left_arm: boneIdx(["l_upper_arm", "l_forearm"]),
  right_arm: boneIdx(["r_upper_arm", "r_forearm"]),
  left_leg: boneIdx(["l_thigh", "l_shin"]),
  right_leg: boneIdx(["r_thigh", "r_shin"]),
  torso: boneIdx(["shoulders", "hips", "l_side", "r_side"]),
};

function boneIdx(keys: string[]): number[] {
  return keys.map((k) => BONES.findIndex((b) => b.key === k)).filter((i) => i >= 0);
}

/** Per-limb error for a single frame: angular error + signed offset. */
interface LimbFrame {
  /** Mean bone-direction error in degrees, or null when no bone was shared. */
  degrees: number | null;
  /** Mean signed test−ref offset (torso units), or null when no shared joint. */
  dx: number | null;
  dy: number | null;
}

/**
 * Per-limb metrics for one ref/test frame.
 *
 * Angular error comes from bone directions (the same viewpoint-robust feature
 * the score uses); the signed offset is the mean displacement of the limb's
 * keypoints in the shared normalized frame, which gives a "which way" hint.
 */
export function frameLimbMetrics(
  ref: NormalizedPose,
  test: NormalizedPose,
): Record<string, LimbFrame> {
  const rv = boneVectors(ref);
  const tv = boneVectors(test);
  const out: Record<string, LimbFrame> = {};
  for (const g of LIMB_GROUPS) {
    // Angular error across the limb's bones.
    let degSum = 0;
    let degCnt = 0;
    for (const bi of LIMB_BONES[g.key] ?? []) {
      const r = rv[bi];
      const t = tv[bi];
      if (!r.valid || !t.valid) continue;
      let cos = r.x * t.x + r.y * t.y;
      if (cos > 1) cos = 1;
      else if (cos < -1) cos = -1;
      degSum += Math.acos(cos) * RAD2DEG;
      degCnt++;
    }
    // Signed offset across the limb's keypoints (test − ref).
    let sx = 0;
    let sy = 0;
    let sCnt = 0;
    for (const name of g.joints) {
      const idx = KEYPOINT_INDEX[name];
      const p = ref.points[idx];
      const q = test.points[idx];
      if (!p || !q || !p.valid || !q.valid) continue;
      sx += q.x - p.x;
      sy += q.y - p.y;
      sCnt++;
    }
    out[g.key] = {
      degrees: degCnt ? degSum / degCnt : null,
      dx: sCnt ? sx / sCnt : null,
      dy: sCnt ? sy / sCnt : null,
    };
  }
  return out;
}

interface FrameRecord {
  tMs: number;
  score: number;
  limbs: Record<string, LimbFrame>;
}

/** Per-limb running accumulator. */
interface LimbAcc {
  degSum: number;
  degCnt: number;
  dxSum: number;
  dySum: number;
  offCnt: number;
}

function newLimbAcc(): LimbAcc {
  return { degSum: 0, degCnt: 0, dxSum: 0, dySum: 0, offCnt: 0 };
}

function accumulate(acc: LimbAcc, f: LimbFrame): void {
  if (f.degrees !== null) {
    acc.degSum += f.degrees;
    acc.degCnt++;
  }
  if (f.dx !== null && f.dy !== null) {
    acc.dxSum += f.dx;
    acc.dySum += f.dy;
    acc.offCnt++;
  }
}

const LABELS: Record<string, string> = Object.fromEntries(
  LIMB_GROUPS.map((g) => [g.key, g.label]),
);

// Below this angular error a limb is "on target"; below this offset magnitude a
// direction component is dropped so we don't nag about sub-noise wobble.
const ON_TARGET_DEG = 8;
const OFFSET_DEADZONE = 0.06;

/**
 * Turn an accumulator into a LimbCorrection, or null when the limb never had a
 * usable angular reading.
 */
function toCorrection(key: string, acc: LimbAcc): LimbCorrection | null {
  if (acc.degCnt === 0) return null;
  const degrees = acc.degSum / acc.degCnt;
  // Correction = move opposite to the test−ref offset.
  const dx = acc.offCnt ? -acc.dxSum / acc.offCnt : 0;
  const dy = acc.offCnt ? -acc.dySum / acc.offCnt : 0;
  return {
    key,
    label: LABELS[key] ?? key,
    degrees,
    dx,
    dy,
    direction: phraseFor(dx, dy),
  };
}

/** Human nudge from a signed correction vector (+dx = right, +dy = down). */
function phraseFor(dx: number, dy: number): string {
  const parts: string[] = [];
  if (dy <= -OFFSET_DEADZONE) parts.push("lift");
  else if (dy >= OFFSET_DEADZONE) parts.push("lower");
  if (dx <= -OFFSET_DEADZONE) parts.push("move screen-left");
  else if (dx >= OFFSET_DEADZONE) parts.push("move screen-right");
  if (parts.length === 0) return "match the reference angle";
  return parts.join(" and ");
}

const round1 = (n: number) => Math.round(n * 10) / 10;

/**
 * Accumulates scored frames over a run and builds the compact CoachingInput.
 *
 * The app pushes one record per analysed frame; `build()` segments the timeline,
 * computes per-limb degrees + signed corrections per segment and overall, and
 * down-samples the score series. Pure data in, pure data out — no DOM, no model.
 */
export class RunReport {
  private frames: FrameRecord[] = [];

  /** Number of frames recorded so far. */
  get size(): number {
    return this.frames.length;
  }

  push(tMs: number, score: number, limbs: Record<string, LimbFrame>): void {
    this.frames.push({ tMs, score, limbs });
  }

  reset(): void {
    this.frames = [];
  }

  /**
   * Build the structured report. `segmentCount` caps how many time slices the
   * routine is divided into; `seriesPoints` caps the down-sampled timeline.
   */
  build(segmentCount = 6, seriesPoints = 24): CoachingInput {
    const frames = this.frames;
    const n = frames.length;
    if (n === 0) {
      return {
        durationSec: 0,
        frames: 0,
        avgScore: 0,
        lowest: null,
        segments: [],
        opportunities: [],
        scoreSeries: [],
      };
    }

    const t0 = frames[0].tMs;
    const tEnd = frames[n - 1].tMs;
    const durationMs = Math.max(0, tEnd - t0);

    // Overall stats + per-limb accumulator.
    let scoreSum = 0;
    let lowest = { score: frames[0].score, t: 0 };
    const overall: Record<string, LimbAcc> = {};
    for (const g of LIMB_GROUPS) overall[g.key] = newLimbAcc();
    for (const f of frames) {
      scoreSum += f.score;
      if (f.score < lowest.score) {
        lowest = { score: f.score, t: round1((f.tMs - t0) / 1000) };
      }
      for (const g of LIMB_GROUPS) accumulate(overall[g.key], f.limbs[g.key]);
    }

    // Segment the timeline into equal time buckets (skip empty ones).
    const buckets = Math.max(1, Math.min(segmentCount, n));
    const span = durationMs > 0 ? durationMs : 1;
    const segAcc: Array<{
      score: number;
      cnt: number;
      limbs: Record<string, LimbAcc>;
    }> = Array.from({ length: buckets }, () => ({
      score: 0,
      cnt: 0,
      limbs: Object.fromEntries(LIMB_GROUPS.map((g) => [g.key, newLimbAcc()])),
    }));
    for (const f of frames) {
      let bi = Math.floor(((f.tMs - t0) / span) * buckets);
      if (bi >= buckets) bi = buckets - 1;
      if (bi < 0) bi = 0;
      const b = segAcc[bi];
      b.score += f.score;
      b.cnt++;
      for (const g of LIMB_GROUPS) accumulate(b.limbs[g.key], f.limbs[g.key]);
    }

    const segments: ReportSegment[] = [];
    let outIdx = 0;
    for (let i = 0; i < buckets; i++) {
      const b = segAcc[i];
      if (b.cnt === 0) continue;
      const startSec = round1((i / buckets) * (durationMs / 1000));
      const endSec = round1(((i + 1) / buckets) * (durationMs / 1000));
      // Worst limb in this segment by mean degrees.
      let worst: LimbCorrection | null = null;
      for (const g of LIMB_GROUPS) {
        const c = toCorrection(g.key, b.limbs[g.key]);
        if (!c) continue;
        if (!worst || c.degrees > worst.degrees) worst = c;
      }
      if (worst) worst = { ...worst, degrees: round1(worst.degrees) };
      segments.push({
        index: outIdx++,
        startSec,
        endSec,
        avgScore: round1(b.score / b.cnt),
        worstLimb: worst,
      });
    }

    // Top opportunities: limbs above the on-target threshold, worst first.
    const opportunities = LIMB_GROUPS.map((g) => toCorrection(g.key, overall[g.key]))
      .filter((c): c is LimbCorrection => c !== null && c.degrees > ON_TARGET_DEG)
      .sort((a, b) => b.degrees - a.degrees)
      .slice(0, 3)
      .map((c) => ({
        ...c,
        degrees: round1(c.degrees),
        dx: round1(c.dx),
        dy: round1(c.dy),
      }));

    // Down-sample the score timeline to at most `seriesPoints` points.
    const stride = Math.max(1, Math.ceil(n / seriesPoints));
    const scoreSeries: ScorePoint[] = [];
    for (let i = 0; i < n; i += stride) {
      scoreSeries.push({
        t: round1((frames[i].tMs - t0) / 1000),
        score: round1(frames[i].score),
      });
    }

    return {
      durationSec: round1(durationMs / 1000),
      frames: n,
      avgScore: round1(scoreSum / n),
      lowest: { score: round1(lowest.score), t: lowest.t },
      segments,
      opportunities,
      scoreSeries,
    };
  }
}
