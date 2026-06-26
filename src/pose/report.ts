// Post-run improvement report (issue #9).
//
// The live score and per-limb breakdown say how a dancer is doing *right now*,
// but tell them nothing actionable once the routine ends. This module records a
// per-bone error series over the whole aligned run, buckets it into time
// segments, ranks the worst limb in each segment, surfaces the biggest
// improvement opportunities across the run, and — crucially — turns each into a
// plain, *directional* correction ("raise the elbow", "level the shoulders").
//
// It builds on the same viewpoint-robust bone-direction feature as scoring
// (boneAngles.ts): per bone we compare the reference and test unit-direction
// vectors. The *magnitude* of their disagreement (the angle between them, in
// degrees) is the error; the *signed* average difference of the unit vectors
// gives the direction to correct in. Because the feature is built from unit
// vectors it cancels scale and translation outright, so a fast, big-movement
// segment is not penalized over a slow, small one — answering the issue's
// normalization concern.

import { boneVectors, BONES } from "./boneAngles";
import type { NormalizedPose } from "./normalize";

/** One bone's reference vs test unit-direction vectors for a single frame. */
export interface BoneSample {
  /** Reference unit-vector x/y (0 when invalid). */
  rx: number;
  ry: number;
  /** Test unit-vector x/y (0 when invalid). */
  tx: number;
  ty: number;
  /** True only when the bone was valid (comparable) in *both* poses. */
  valid: boolean;
}

/** All bones for one analysed frame, plus its aligned reference time. */
export interface FrameSample {
  /** Aligned reference-timeline time in seconds (the canonical run clock). */
  t: number;
  /** Per-bone vectors, index-aligned with BONES. */
  bones: BoneSample[];
}

/** How a limb did within one segment, with directional coaching. */
export interface LimbSegmentScore {
  limbKey: string;
  limbLabel: string;
  /** Mean angular error across this limb's bones & frames, in degrees. */
  meanErrorDeg: number;
  /** How many bone×frame errors fed the mean (0 → limb not seen here). */
  samples: number;
  /** Bone (key) that diverged most — the one the correction targets. */
  dominantBoneKey: string | null;
  /** Plain, directional fix ("Raise your left elbow — it's ~25° too low."). */
  correction: string;
}

/** One time bucket of the routine and how each limb fared in it. */
export interface SegmentReport {
  index: number;
  startSec: number;
  endSec: number;
  /** Human time range, e.g. "0:42–0:55". */
  label: string;
  /** Frames that fell into this segment. */
  frameCount: number;
  /** All limbs seen in this segment, sorted worst (highest error) first. */
  limbs: LimbSegmentScore[];
  /** Convenience pointer to limbs[0] (the worst limb), or null if none. */
  worst: LimbSegmentScore | null;
}

/** A limb×segment pair ranked as a run-wide improvement opportunity. */
export interface Opportunity extends LimbSegmentScore {
  segmentIndex: number;
  startSec: number;
  endSec: number;
  label: string;
}

export interface ImprovementReport {
  segments: SegmentReport[];
  /** Highest-error limb×segment pairs across the whole run, worst first. */
  topOpportunities: Opportunity[];
  /** Length of the recorded run, in seconds. */
  durationSec: number;
  /** Total frames recorded. */
  frameCount: number;
}

export interface ReportOptions {
  /** Segment width in seconds (fixed-window segmentation). Default 3. */
  segmentSec?: number;
  /** How many run-wide opportunities to surface. Default 5. */
  topN?: number;
}

const DEFAULT_SEGMENT_SEC = 3;
const DEFAULT_TOP_N = 5;
// Below this mean error a limb is "on the reference" and not worth coaching, so
// it is excluded from the worst-of and opportunity lists.
const MIN_REPORTABLE_DEG = 6;

/** Which bones make up each reportable limb (mirrors perJoint.ts groupings). */
const LIMB_BONES: Array<{ key: string; label: string; bones: string[] }> = [
  { key: "left_arm", label: "Left arm", bones: ["l_upper_arm", "l_forearm"] },
  { key: "right_arm", label: "Right arm", bones: ["r_upper_arm", "r_forearm"] },
  { key: "left_leg", label: "Left leg", bones: ["l_thigh", "l_shin"] },
  { key: "right_leg", label: "Right leg", bones: ["r_thigh", "r_shin"] },
  {
    key: "torso",
    label: "Torso",
    bones: ["shoulders", "hips", "l_side", "r_side"],
  },
];

const BONE_INDEX: Record<string, number> = BONES.reduce(
  (acc, b, i) => {
    acc[b.key] = i;
    return acc;
  },
  {} as Record<string, number>,
);

/**
 * Accumulates the per-bone reference-vs-test direction series over a run. Push
 * one analysed frame at a time; call build() at the end for the report.
 */
export class RunRecorder {
  private samples: FrameSample[] = [];

  /**
   * Record one analysed frame. `t` is the aligned reference time (seconds);
   * `ref` is the (possibly DTW/live-sync matched) reference pose actually
   * scored against `test`.
   */
  push(t: number, ref: NormalizedPose, test: NormalizedPose): void {
    const rv = boneVectors(ref);
    const tv = boneVectors(test);
    const bones: BoneSample[] = rv.map((r, i) => {
      const tvec = tv[i];
      return {
        rx: r.x,
        ry: r.y,
        tx: tvec.x,
        ty: tvec.y,
        valid: r.valid && tvec.valid,
      };
    });
    this.samples.push({ t, bones });
  }

  reset(): void {
    this.samples = [];
  }

  get length(): number {
    return this.samples.length;
  }

  build(opts?: ReportOptions): ImprovementReport {
    return buildReport(this.samples, opts);
  }
}

/** Format seconds as m:ss (e.g. 42 → "0:42", 75 → "1:15"). */
export function formatTime(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, "0")}`;
}

/** Angle (degrees) between two unit vectors; 0 when either is degenerate. */
function angleDeg(ax: number, ay: number, bx: number, by: number): number {
  let cos = ax * bx + ay * by;
  if (cos > 1) cos = 1;
  else if (cos < -1) cos = -1;
  return (Math.acos(cos) * 180) / Math.PI;
}

interface BoneAccum {
  errSum: number;
  errCnt: number;
  /** Averaged signed difference of unit vectors (test − ref). */
  ddx: number;
  ddy: number;
}

/**
 * Translate a bone's averaged direction error into a concrete, directional
 * correction. We pick the dominant axis of the signed unit-vector delta
 * (test − ref) and phrase a move that brings the dancer back toward the
 * reference. Image coordinates: +y is *down*, +x is *right*.
 */
function correctionFor(boneKey: string, ddx: number, ddy: number, errDeg: number): string {
  const deg = Math.round(errDeg);
  const off = `~${deg}° off`;
  const vertical = Math.abs(ddy) >= Math.abs(ddx);

  // Torso bones read as tilt/lean, not as a single moved joint.
  switch (boneKey) {
    case "shoulders":
      return vertical
        ? `Level your shoulders — one side is dropped (${off}).`
        : `Square your shoulders to the camera (${off}).`;
    case "hips":
      return vertical
        ? `Level your hips — your weight is tilted to one side (${off}).`
        : `Square your hips to the camera (${off}).`;
    case "l_side":
    case "r_side": {
      const side = boneKey === "l_side" ? "left" : "right";
      return vertical
        ? `Lengthen your ${side} side — you're collapsing into it (${off}).`
        : `Straighten your torso — you're leaning ${ddx > 0 ? "right" : "left"} (${off}).`;
    }
  }

  // Limb bones: name the moving (child) joint and give a raise/lower/shift cue.
  const joint = LIMB_JOINT[boneKey] ?? boneKey;
  if (vertical) {
    return ddy > 0
      ? `Raise your ${joint} — it's ${off}, dropping too low.`
      : `Lower your ${joint} — it's ${off}, riding too high.`;
  }
  return ddx > 0
    ? `Bring your ${joint} left — it's drifting right (${off}).`
    : `Bring your ${joint} right — it's drifting left (${off}).`;
}

/** Natural name of the moving (child) joint for each limb bone. */
const LIMB_JOINT: Record<string, string> = {
  l_upper_arm: "left elbow",
  l_forearm: "left wrist",
  r_upper_arm: "right elbow",
  r_forearm: "right wrist",
  l_thigh: "left knee",
  l_shin: "left ankle",
  r_thigh: "right knee",
  r_shin: "right ankle",
};

/** Score every limb within a set of frames (one segment). */
function scoreLimbs(frames: FrameSample[]): LimbSegmentScore[] {
  // Accumulate per-bone error + signed direction delta across the frames.
  const accum = new Map<string, BoneAccum>();
  for (const f of frames) {
    for (const lb of LIMB_BONES) {
      for (const boneKey of lb.bones) {
        const bi = BONE_INDEX[boneKey];
        const s = f.bones[bi];
        if (!s || !s.valid) continue;
        const err = angleDeg(s.rx, s.ry, s.tx, s.ty);
        let a = accum.get(boneKey);
        if (!a) {
          a = { errSum: 0, errCnt: 0, ddx: 0, ddy: 0 };
          accum.set(boneKey, a);
        }
        a.errSum += err;
        a.errCnt += 1;
        a.ddx += s.tx - s.rx;
        a.ddy += s.ty - s.ry;
      }
    }
  }

  const limbs: LimbSegmentScore[] = [];
  for (const lb of LIMB_BONES) {
    let errSum = 0;
    let errCnt = 0;
    let dominant: { key: string; mean: number; ddx: number; ddy: number } | null = null;
    for (const boneKey of lb.bones) {
      const a = accum.get(boneKey);
      if (!a || a.errCnt === 0) continue;
      errSum += a.errSum;
      errCnt += a.errCnt;
      const mean = a.errSum / a.errCnt;
      if (!dominant || mean > dominant.mean) {
        dominant = {
          key: boneKey,
          mean,
          ddx: a.ddx / a.errCnt,
          ddy: a.ddy / a.errCnt,
        };
      }
    }
    if (errCnt === 0) {
      limbs.push({
        limbKey: lb.key,
        limbLabel: lb.label,
        meanErrorDeg: 0,
        samples: 0,
        dominantBoneKey: null,
        correction: "Not enough data for this limb in this segment.",
      });
      continue;
    }
    const meanErrorDeg = errSum / errCnt;
    limbs.push({
      limbKey: lb.key,
      limbLabel: lb.label,
      meanErrorDeg,
      samples: errCnt,
      dominantBoneKey: dominant ? dominant.key : null,
      correction: dominant
        ? correctionFor(dominant.key, dominant.ddx, dominant.ddy, dominant.mean)
        : "Keep this limb matched to the reference.",
    });
  }

  // Worst (highest error) first; limbs with no data sink to the bottom.
  limbs.sort((a, b) => b.meanErrorDeg - a.meanErrorDeg);
  return limbs;
}

/**
 * Build the full improvement report from a recorded frame series: bucket into
 * fixed time windows, rank limbs within each, and surface the run-wide top-N
 * limb×segment opportunities.
 */
export function buildReport(
  samples: FrameSample[],
  opts: ReportOptions = {},
): ImprovementReport {
  const segmentSec = Math.max(0.5, opts.segmentSec ?? DEFAULT_SEGMENT_SEC);
  const topN = Math.max(1, opts.topN ?? DEFAULT_TOP_N);

  if (samples.length === 0) {
    return { segments: [], topOpportunities: [], durationSec: 0, frameCount: 0 };
  }

  let maxT = 0;
  for (const s of samples) if (s.t > maxT) maxT = s.t;
  const durationSec = maxT;

  const nSeg = Math.max(1, Math.ceil((durationSec + 1e-6) / segmentSec));
  const segments: SegmentReport[] = [];
  const opportunities: Opportunity[] = [];

  for (let i = 0; i < nSeg; i++) {
    const startSec = i * segmentSec;
    const endSec = Math.min(durationSec, (i + 1) * segmentSec);
    // Last segment is inclusive of the final timestamp so no frame is dropped.
    const isLast = i === nSeg - 1;
    const frames = samples.filter(
      (s) => s.t >= startSec && (isLast ? s.t <= endSec + 1e-6 : s.t < (i + 1) * segmentSec),
    );
    if (frames.length === 0) continue;

    const limbs = scoreLimbs(frames);
    const reportable = limbs.filter((l) => l.samples > 0 && l.meanErrorDeg >= MIN_REPORTABLE_DEG);
    const worst = reportable.length > 0 ? reportable[0] : null;
    const label = `${formatTime(startSec)}–${formatTime(endSec)}`;

    segments.push({
      index: i,
      startSec,
      endSec,
      label,
      frameCount: frames.length,
      limbs,
      worst,
    });

    for (const l of reportable) {
      opportunities.push({
        ...l,
        segmentIndex: i,
        startSec,
        endSec,
        label,
      });
    }
  }

  opportunities.sort((a, b) => b.meanErrorDeg - a.meanErrorDeg);
  const topOpportunities = opportunities.slice(0, topN);

  return {
    segments,
    topOpportunities,
    durationSec,
    frameCount: samples.length,
  };
}
