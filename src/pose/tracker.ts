// Light multi-person tracking + single-target lock.
//
// Per-frame pose detection has no memory: from one frame to the next the
// detector returns an unordered set of people with no notion of "the same
// body." When a clip contains more than one figure, downstream scoring needs a
// stable identity so the skeleton (and the similarity score) follows ONE dancer
// instead of flickering between whoever the model ranks highest that frame.
//
// This module is deliberately framework-free — no TF.js, no DOM — so the
// matching logic is unit-testable headless (see demo/verify.mjs `verifyTracker`)
// and reused identically by both the single- and multi-person UI paths.
//
//   PoseTracker  — assigns stable integer ids to detections across frames by
//                  greedy matching on bbox IoU + normalized keypoint distance,
//                  coasting briefly through short dropouts/occlusions.
//   TargetLock   — follows one chosen id; on a dropout it holds the last pose
//                  and re-acquires the SAME body within a gate rather than
//                  snapping to a random person.

import { type Pose } from "./keypoints";
import { DEFAULT_MIN_KEYPOINT_SCORE } from "./normalize";

// Augment the shared Pose with an OPTIONAL appearance fingerprint so the
// detector can attach a torso color histogram and the tracker can read it,
// without touching keypoints.ts. 2D-only / Node paths leave it undefined and
// the tracker falls back to motion-only matching (identical to old behavior).
declare module "./keypoints" {
  interface Pose {
    /** Normalized HSV torso color histogram (HIST_BINS long); optional. */
    appearance?: number[];
  }
}

// ---------------------------------------------------------------------------
// Appearance cue (DeepSORT-lite): HSV torso color histogram + Bhattacharyya
// distance. Pure & deterministic — operates on plain number arrays so it is
// unit-testable headless and reusable in the browser identically.
// ---------------------------------------------------------------------------

/** Hue bins × Saturation bins for the torso color histogram. */
export const HIST_HUE_BINS = 8;
export const HIST_SAT_BINS = 4;
export const HIST_BINS = HIST_HUE_BINS * HIST_SAT_BINS; // 32

// Pixels too dark, too bright, or too gray carry no reliable hue and are gated
// out so shadows / highlights / skin-ish neutrals don't pollute the fingerprint.
const MIN_VALUE = 0.15; // gate near-black
const MAX_VALUE = 0.95; // gate near-white
const MIN_SAT = 0.2; // gate low-saturation (gray) pixels

/**
 * Convert an 8-bit RGB triple to HSV, each channel in [0, 1).
 * Hue wraps in [0, 1); for achromatic input hue is 0 and saturation 0.
 */
export function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const rn = r / 255,
    gn = g / 255,
    bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;
  let h = 0;
  if (delta > 1e-9) {
    if (max === rn) h = ((gn - bn) / delta) % 6;
    else if (max === gn) h = (bn - rn) / delta + 2;
    else h = (rn - gn) / delta + 4;
    h /= 6;
    if (h < 0) h += 1;
  }
  const s = max <= 0 ? 0 : delta / max;
  const v = max;
  return [h, s, v];
}

/**
 * Build a normalized hue×saturation histogram from a flat RGB pixel buffer
 * ([r,g,b, r,g,b, ...]). Near-black/near-white/low-sat pixels are gated out.
 * Returns a HIST_BINS-long array summing to 1 (or all zeros if no pixel
 * qualified). Pure: takes plain numbers, no canvas/DOM.
 */
export function buildHistogram(rgb: ArrayLike<number>): number[] {
  const hist = new Array(HIST_BINS).fill(0);
  let total = 0;
  for (let i = 0; i + 2 < rgb.length; i += 3) {
    const [h, s, v] = rgbToHsv(rgb[i], rgb[i + 1], rgb[i + 2]);
    if (v < MIN_VALUE || v > MAX_VALUE || s < MIN_SAT) continue;
    let hb = Math.floor(h * HIST_HUE_BINS);
    if (hb >= HIST_HUE_BINS) hb = HIST_HUE_BINS - 1;
    let sb = Math.floor(s * HIST_SAT_BINS);
    if (sb >= HIST_SAT_BINS) sb = HIST_SAT_BINS - 1;
    hist[hb * HIST_SAT_BINS + sb]++;
    total++;
  }
  if (total > 0) for (let i = 0; i < hist.length; i++) hist[i] /= total;
  return hist;
}

/**
 * Symmetric Bhattacharyya distance between two normalized histograms, in
 * [0, 1]: 0 = identical, 1 = no overlap. Mismatched/empty histograms return 1
 * (max distance — there is no appearance evidence to match on).
 */
export function histogramDistance(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 1;
  let sumA = 0,
    sumB = 0;
  for (let i = 0; i < a.length; i++) {
    sumA += a[i];
    sumB += b[i];
  }
  if (sumA <= 0 || sumB <= 0) return 1;
  let bc = 0; // Bhattacharyya coefficient over re-normalized histograms
  for (let i = 0; i < a.length; i++) bc += Math.sqrt((a[i] / sumA) * (b[i] / sumB));
  bc = Math.min(1, Math.max(0, bc));
  // 1 - BC is already in [0,1] and symmetric; preferred over sqrt(1-BC) here so
  // identical histograms read ~0 and disjoint ones ~1.
  return 1 - bc;
}

/**
 * EMA-blend a track's stored appearance toward a fresh observation. With no
 * prior, adopt the observation; with no observation, keep the prior.
 */
function blendAppearance(
  prior: number[] | undefined,
  obs: number[] | undefined,
  alpha: number,
): number[] | undefined {
  if (!obs || obs.length === 0) return prior;
  if (!prior || prior.length !== obs.length) return obs.slice();
  const out = new Array(obs.length);
  for (let i = 0; i < obs.length; i++) out[i] = prior[i] * (1 - alpha) + obs[i] * alpha;
  return out;
}

/**
 * Hungarian (Kuhn–Munkres) optimal assignment minimizing total cost over a
 * rectangular `rows × cols` matrix. Returns `assign[r]` = the column matched to
 * row r, or -1 if row r is left unmatched. Implementation pads to a square
 * matrix internally; pads use `pad` as a neutral-high filler so real rows are
 * preferentially matched. Pure & deterministic.
 */
function hungarian(cost: number[][], pad: number): number[] {
  const nRows = cost.length;
  const nCols = nRows > 0 ? cost[0].length : 0;
  if (nRows === 0 || nCols === 0) return new Array(nRows).fill(-1);
  const n = Math.max(nRows, nCols);

  // Square cost matrix (1-indexed for the classic O(n^3) potentials method).
  const a: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(n);
    for (let j = 0; j < n; j++) {
      row[j] = i < nRows && j < nCols ? cost[i][j] : pad;
    }
    a.push(row);
  }

  const INF = Number.POSITIVE_INFINITY;
  const u = new Array(n + 1).fill(0);
  const v = new Array(n + 1).fill(0);
  const p = new Array(n + 1).fill(0); // p[j] = row assigned to column j
  const way = new Array(n + 1).fill(0);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(n + 1).fill(INF);
    const used = new Array(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF;
      let j1 = -1;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = a[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  const assign = new Array(nRows).fill(-1);
  for (let j = 1; j <= n; j++) {
    const i = p[j];
    if (i >= 1 && i <= nRows && j <= nCols) assign[i - 1] = j - 1;
  }
  return assign;
}

/** Axis-aligned bounding box in source-video pixel space. */
export interface BBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** A detection with the stable id the tracker assigned it this frame. */
export interface TrackedPose {
  id: number;
  pose: Pose;
  bbox: BBox;
}

/** Bounding box over a pose's confident keypoints, or null if none qualify. */
export function poseBBox(pose: Pose, minScore = DEFAULT_MIN_KEYPOINT_SCORE): BBox | null {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity,
    n = 0;
  for (const k of pose.keypoints) {
    if ((k.score ?? 0) < minScore) continue;
    minX = Math.min(minX, k.x);
    minY = Math.min(minY, k.y);
    maxX = Math.max(maxX, k.x);
    maxY = Math.max(maxY, k.y);
    n++;
  }
  if (n === 0 || maxX < minX || maxY < minY) return null;
  return { minX, minY, maxX, maxY };
}

export function bboxArea(b: BBox): number {
  return Math.max(0, b.maxX - b.minX) * Math.max(0, b.maxY - b.minY);
}

export function bboxCenter(b: BBox): { x: number; y: number } {
  return { x: (b.minX + b.maxX) / 2, y: (b.minY + b.maxY) / 2 };
}

export function bboxDiag(b: BBox): number {
  return Math.hypot(b.maxX - b.minX, b.maxY - b.minY);
}

export function bboxIoU(a: BBox, b: BBox): number {
  const ix = Math.max(0, Math.min(a.maxX, b.maxX) - Math.max(a.minX, b.minX));
  const iy = Math.max(0, Math.min(a.maxY, b.maxY) - Math.max(a.minY, b.minY));
  const inter = ix * iy;
  const uni = bboxArea(a) + bboxArea(b) - inter;
  return uni <= 0 ? 0 : inter / uni;
}

/**
 * Mean Euclidean distance over keypoints both poses detect confidently,
 * normalized by the mean bbox diagonal so the term is scale-invariant. Returns
 * null when the two poses share no confident keypoints.
 */
export function normKeypointDistance(
  a: Pose,
  b: Pose,
  aBox: BBox,
  bBox: BBox,
  minScore = DEFAULT_MIN_KEYPOINT_SCORE,
): number | null {
  const ka = a.keypoints;
  const kb = b.keypoints;
  const n = Math.min(ka.length, kb.length);
  let sum = 0,
    cnt = 0;
  for (let i = 0; i < n; i++) {
    if ((ka[i].score ?? 0) < minScore || (kb[i].score ?? 0) < minScore) continue;
    sum += Math.hypot(ka[i].x - kb[i].x, ka[i].y - kb[i].y);
    cnt++;
  }
  if (cnt === 0) return null;
  const scale = (bboxDiag(aBox) + bboxDiag(bBox)) / 2 || 1;
  return sum / cnt / scale;
}

export interface TrackerOptions {
  /** Weight on the IoU term (1 - IoU) in the match cost. */
  iouWeight?: number;
  /** Weight on the normalized-keypoint-distance term in the match cost. */
  distWeight?: number;
  /** Pairs costing more than this are never matched (a new id is spawned). */
  maxCost?: number;
  /** Frames a track may coast unmatched before it is dropped. */
  maxAge?: number;
  /**
   * Weight on the motion term `(1 - IoU(predictedBox, detBox))` in the combined
   * cost used by the global (Hungarian) assignment.
   */
  motionWeight?: number;
  /** Weight on the appearance (histogram) term in the combined cost. */
  appearanceWeight?: number;
  /**
   * Motion gate: a (track, detection) pair whose center distance exceeds
   * `gate * bboxDiag` is infeasible regardless of appearance.
   */
  gate?: number;
  /** EMA factor for blending a track's appearance toward each observation. */
  appearanceEma?: number;
  /** EMA factor for the constant-velocity estimate. */
  velocityEma?: number;
}

interface Track {
  id: number;
  pose: Pose;
  bbox: BBox;
  age: number; // frames since last matched (0 when matched this frame)
  hits: number;
  appearance?: number[]; // EMA-blended torso color histogram
  vx: number; // constant-velocity estimate of bbox-center motion (px/frame)
  vy: number;
}

/**
 * Greedy multi-target tracker. Each `update(poses)` matches this frame's
 * detections to existing tracks by ascending cost, spawns ids for the leftovers,
 * and ages/drops tracks that go unseen — so a body that briefly disappears keeps
 * its id when it returns (within `maxAge`).
 */
export class PoseTracker {
  private tracks: Track[] = [];
  private nextId = 1;
  private readonly iouW: number;
  private readonly distW: number;
  private readonly maxCost: number;
  private readonly maxAge: number;
  private readonly motionW: number;
  private readonly appearW: number;
  private readonly gate: number;
  private readonly appearEma: number;
  private readonly velEma: number;

  constructor(opts: TrackerOptions = {}) {
    this.iouW = opts.iouWeight ?? 0.6;
    this.distW = opts.distWeight ?? 0.4;
    this.maxCost = opts.maxCost ?? 0.75;
    this.maxAge = opts.maxAge ?? 30;
    this.motionW = opts.motionWeight ?? 0.6;
    this.appearW = opts.appearanceWeight ?? 0.4;
    this.gate = opts.gate ?? 1.5;
    this.appearEma = opts.appearanceEma ?? 0.25;
    this.velEma = opts.velocityEma ?? 0.5;
  }

  reset(): void {
    this.tracks = [];
    this.nextId = 1;
  }

  /** Number of live tracks (including ones currently coasting). */
  get trackCount(): number {
    return this.tracks.length;
  }

  /**
   * Constant-velocity prediction of a track's box one frame ahead. A track that
   * missed `age` frames is propagated `age` frames along its velocity estimate
   * so its predicted box still overlaps where the body should now be.
   */
  private predictBox(track: Track): BBox {
    const steps = Math.max(1, track.age);
    const dx = track.vx * steps;
    const dy = track.vy * steps;
    return {
      minX: track.bbox.minX + dx,
      minY: track.bbox.minY + dy,
      maxX: track.bbox.maxX + dx,
      maxY: track.bbox.maxY + dy,
    };
  }

  /**
   * Combined assignment cost for (track, detection): a motion term over the
   * track's PREDICTED box plus an appearance (color-histogram) term. Returns
   * `Infinity` when the motion gate rejects the pair so it can never match.
   * Kept low-level so the Hungarian solver can build a full cost matrix.
   */
  private cost(track: Track, box: BBox, pose: Pose): number {
    const predicted = this.predictBox(track);

    // Motion gate: reject pairs whose centers are farther than gate * diag.
    const pc = bboxCenter(predicted);
    const bc = bboxCenter(box);
    const norm = bboxDiag(predicted) || bboxDiag(track.bbox) || 1;
    if (Math.hypot(pc.x - bc.x, pc.y - bc.y) > this.gate * norm) return Infinity;

    // Motion term over the predicted box (IoU + normalized keypoint distance).
    const iou = bboxIoU(predicted, box);
    const nd = normKeypointDistance(track.pose, pose, predicted, box);
    const motion = this.iouW * (1 - iou) + this.distW * Math.min(1, nd ?? 1);

    // Appearance term: Bhattacharyya distance when both sides have a histogram;
    // otherwise neutral (0.5) so motion drives the decision (no penalty/bonus).
    const obs = pose.appearance;
    const hist =
      track.appearance && track.appearance.length && obs && obs.length
        ? histogramDistance(track.appearance, obs)
        : 0.5;

    return this.motionW * motion + this.appearW * hist;
  }

  /** Ingest a frame's detections; return them tagged with stable ids. */
  update(poses: Pose[]): TrackedPose[] {
    const boxes = poses.map((p) => poseBBox(p));

    const nT = this.tracks.length;
    const validD: number[] = [];
    for (let d = 0; d < poses.length; d++) if (boxes[d]) validD.push(d);

    // Build the full track×detection cost matrix; gated/over-budget pairs are
    // BIG (effectively forbidden) so the global assignment never picks them.
    const BIG = 1e6;
    const costM: number[][] = [];
    for (let t = 0; t < nT; t++) {
      const row: number[] = [];
      for (let j = 0; j < validD.length; j++) {
        const d = validD[j];
        const c = this.cost(this.tracks[t], boxes[d]!, poses[d]);
        row.push(c <= this.maxCost ? c : BIG);
      }
      costM.push(row);
    }

    // Global (Hungarian / Kuhn-Munkres) optimal assignment over the matrix.
    const assign = hungarian(costM, BIG);

    const usedT = new Set<number>();
    const usedD = new Set<number>();
    const out: Array<TrackedPose | undefined> = new Array(poses.length);

    for (let t = 0; t < nT; t++) {
      const j = assign[t];
      if (j < 0 || j >= validD.length) continue;
      if (costM[t][j] >= BIG) continue; // forbidden pair → leave unmatched
      const d = validD[j];
      usedT.add(t);
      usedD.add(d);
      const tr = this.tracks[t];
      const box = boxes[d]!;

      // Update constant-velocity estimate from box-center displacement.
      const prevC = bboxCenter(tr.bbox);
      const newC = bboxCenter(box);
      const steps = Math.max(1, tr.age);
      const mvx = (newC.x - prevC.x) / steps;
      const mvy = (newC.y - prevC.y) / steps;
      tr.vx = tr.vx * (1 - this.velEma) + mvx * this.velEma;
      tr.vy = tr.vy * (1 - this.velEma) + mvy * this.velEma;

      tr.appearance = blendAppearance(tr.appearance, poses[d].appearance, this.appearEma);
      tr.pose = poses[d];
      tr.bbox = box;
      tr.age = 0;
      tr.hits++;
      out[d] = { id: tr.id, pose: poses[d], bbox: box };
    }

    // Age unmatched existing tracks; drop the stale ones. Do this BEFORE adding
    // new tracks so freshly-spawned ids aren't mistaken for unmatched.
    const survivors: Track[] = [];
    for (let t = 0; t < nT; t++) {
      const tr = this.tracks[t];
      if (!usedT.has(t)) {
        tr.age++;
        if (tr.age > this.maxAge) continue; // drop
      }
      survivors.push(tr);
    }
    this.tracks = survivors;

    // Unmatched detections become new tracks.
    for (let d = 0; d < poses.length; d++) {
      if (usedD.has(d)) continue;
      const box = boxes[d];
      if (!box) continue; // skip poses with no usable bbox
      const tr: Track = {
        id: this.nextId++,
        pose: poses[d],
        bbox: box,
        age: 0,
        hits: 1,
        appearance: poses[d].appearance ? poses[d].appearance!.slice() : undefined,
        vx: 0,
        vy: 0,
      };
      this.tracks.push(tr);
      out[d] = { id: tr.id, pose: poses[d], bbox: box };
    }

    return out.filter((x): x is TrackedPose => x !== undefined);
  }
}

export type LockStatus = "searching" | "tracking" | "coasting" | "lost";

export interface LockResult {
  /** Best available pose to display: a live match, or the held last pose. */
  pose: Pose | null;
  /** True only when `pose` is a detection matched live this frame. */
  fresh: boolean;
  /** The locked track id, or null when nothing is locked yet. */
  id: number | null;
  status: LockStatus;
}

export interface SelectOptions {
  /** Auto-pick a target (largest / most-central / most-confident) if unlocked. */
  autoPick?: boolean;
  /** A click point (source-video pixels) to (re)select the nearest person. */
  click?: { x: number; y: number } | null;
  /** Frame size, used to bias auto-pick toward the center. */
  frame?: { width: number; height: number } | null;
  /** Re-acquire radius, in multiples of the last bbox diagonal. */
  reacquireGate?: number;
  /** Frames to coast on the last pose before reporting "lost". */
  maxCoast?: number;
  /**
   * Appearance EMA factor for the lock's own remembered fingerprint (separate
   * from the tracker's per-track EMA).
   */
  appearanceEma?: number;
  /**
   * Max appearance (histogram) distance a re-acquire candidate may have from
   * the lock's remembered color. Beyond this, the candidate is rejected.
   */
  maxRelockAppearance?: number;
  /**
   * Ambiguity guard: if the top-2 re-acquire candidates' appearance distances
   * are within this margin, the situation is ambiguous and the lock returns
   * "lost" rather than hopping to a possible bystander.
   */
  ambiguityMargin?: number;
  /**
   * Raised confidence bar: a re-acquire candidate's pose score must be at least
   * this high (when scores are available) to be eligible.
   */
  relockMinScore?: number;
}

/**
 * Follows one chosen person across frames. When the locked id is absent for a
 * frame it holds the last pose and tries to re-acquire the SAME body within a
 * gate around its last position — so a brief occlusion or detector dropout never
 * snaps the lock onto a different dancer.
 */
export class TargetLock {
  private lockedId: number | null = null;
  private lastPose: Pose | null = null;
  private lastBBox: BBox | null = null;
  private lastAppearance: number[] | undefined = undefined;
  private coast = 0;

  reset(): void {
    this.lockedId = null;
    this.lastPose = null;
    this.lastBBox = null;
    this.lastAppearance = undefined;
    this.coast = 0;
  }

  get id(): number | null {
    return this.lockedId;
  }

  /** Force the lock onto a specific track id. */
  lockTo(id: number): void {
    this.lockedId = id;
    this.coast = 0;
  }

  /**
   * Public relock-by-point: choose the body under (or nearest) a source-pixel
   * point and lock onto it immediately, returning the picked detection (or null
   * if there were no candidates). Used by the live-webcam tap-to-relock handler
   * so it doesn't have to wait a frame for `select({ click })` — and so the tap
   * logic isn't duplicated outside this class.
   */
  pickAt(
    tracked: TrackedPose[],
    pt: { x: number; y: number },
  ): TrackedPose | null {
    const picked = this.pickByPoint(tracked, pt);
    if (picked) this.acquire(picked);
    return picked;
  }

  private pickByPoint(
    tracked: TrackedPose[],
    pt: { x: number; y: number },
  ): TrackedPose | null {
    let best: TrackedPose | null = null;
    let bestD = Infinity;
    for (const t of tracked) {
      const c = bboxCenter(t.bbox);
      const inside =
        pt.x >= t.bbox.minX &&
        pt.x <= t.bbox.maxX &&
        pt.y >= t.bbox.minY &&
        pt.y <= t.bbox.maxY;
      // Prefer a box that contains the click; otherwise the nearest center.
      const d = Math.hypot(pt.x - c.x, pt.y - c.y) / (bboxDiag(t.bbox) || 1) - (inside ? 1 : 0);
      if (d < bestD) {
        bestD = d;
        best = t;
      }
    }
    return best;
  }

  private autoPick(
    tracked: TrackedPose[],
    frame?: { width: number; height: number } | null,
  ): TrackedPose | null {
    let best: TrackedPose | null = null;
    let bestScore = -Infinity;
    const fc = frame ? { x: frame.width / 2, y: frame.height / 2 } : null;
    const fd = frame ? Math.hypot(frame.width, frame.height) : 1;
    for (const t of tracked) {
      const area = bboxArea(t.bbox);
      let central = 1;
      if (fc) {
        const c = bboxCenter(t.bbox);
        central = 1 - Math.min(1, Math.hypot(c.x - fc.x, c.y - fc.y) / (fd / 2));
      }
      const conf = t.pose.score ?? 1;
      // Largest body wins, gently weighted toward the frame center + confidence.
      const s = area * (0.5 + 0.5 * central) * (0.5 + 0.5 * conf);
      if (s > bestScore) {
        bestScore = s;
        best = t;
      }
    }
    return best;
  }

  private acquire(
    picked: TrackedPose,
    status: LockStatus = "tracking",
    appearanceEma = 0.25,
  ): LockResult {
    this.lockedId = picked.id;
    this.coast = 0;
    this.lastPose = picked.pose;
    this.lastBBox = picked.bbox;
    this.lastAppearance = blendAppearance(
      this.lastAppearance,
      picked.pose.appearance,
      appearanceEma,
    );
    return { pose: picked.pose, fresh: true, id: picked.id, status };
  }

  /** Resolve which person to follow this frame from the tracked detections. */
  select(tracked: TrackedPose[], opts: SelectOptions = {}): LockResult {
    const reGate = opts.reacquireGate ?? 1.5;
    const maxCoast = opts.maxCoast ?? 30;
    const appearanceEma = opts.appearanceEma ?? 0.25;
    const maxRelockAppearance = opts.maxRelockAppearance ?? 0.3;
    const ambiguityMargin = opts.ambiguityMargin ?? 0.1;
    const relockMinScore = opts.relockMinScore ?? 0.3;

    // An explicit click always (re)selects, overriding the current lock.
    if (opts.click) {
      const picked = this.pickByPoint(tracked, opts.click);
      if (picked) return this.acquire(picked, "tracking", appearanceEma);
    }

    // Nothing locked yet: auto-pick a target if asked to.
    if (this.lockedId === null) {
      if (opts.autoPick) {
        const picked = this.autoPick(tracked, opts.frame);
        if (picked) return this.acquire(picked, "tracking", appearanceEma);
      }
      return { pose: null, fresh: false, id: null, status: "searching" };
    }

    // Locked id present this frame → clean live match.
    const match = tracked.find((t) => t.id === this.lockedId);
    if (match) return this.acquire(match, "tracking", appearanceEma);

    // Locked id missing: try a hardened re-acquire of the SAME body. A
    // candidate must pass the motion gate, the raised confidence bar, and the
    // appearance bar; and if the two best candidates look equally like the
    // target (within ambiguityMargin) we refuse to hop and report "lost".
    if (this.lastBBox) {
      const c1 = bboxCenter(this.lastBBox);
      const norm = bboxDiag(this.lastBBox) || 1;
      const haveColor = !!(this.lastAppearance && this.lastAppearance.length);

      type Cand = { t: TrackedPose; motion: number; appear: number };
      const cands: Cand[] = [];
      for (const t of tracked) {
        const c2 = bboxCenter(t.bbox);
        const motion = Math.hypot(c1.x - c2.x, c1.y - c2.y) / norm;
        if (motion > reGate) continue; // motion gate
        if ((t.pose.score ?? 1) < relockMinScore) continue; // raised conf bar
        const appear =
          haveColor && t.pose.appearance && t.pose.appearance.length
            ? histogramDistance(this.lastAppearance!, t.pose.appearance)
            : 0.5; // no color evidence → neutral
        // Appearance bar (only meaningful when we actually have color on both).
        if (haveColor && t.pose.appearance && t.pose.appearance.length) {
          if (appear > maxRelockAppearance) continue;
        }
        cands.push({ t, motion, appear });
      }

      if (cands.length === 1) {
        return this.acquire(cands[0].t, "tracking", appearanceEma);
      }
      if (cands.length >= 2) {
        // Rank by appearance first (then motion as the tiebreak) — appearance is
        // what disambiguates a crossing/occlusion.
        cands.sort((x, y) => x.appear - y.appear || x.motion - y.motion);
        const best = cands[0];
        const second = cands[1];
        // Ambiguity guard: if the runner-up is appearance-indistinguishable from
        // the best, do NOT hop — report lost instead of risking a bystander.
        if (haveColor && second.appear - best.appear < ambiguityMargin) {
          this.coast++;
          return { pose: this.lastPose, fresh: false, id: this.lockedId, status: "lost" };
        }
        return this.acquire(best.t, "tracking", appearanceEma);
      }
    }

    // No suitable body: coast on the last known pose for a while.
    this.coast++;
    const status: LockStatus = this.coast <= maxCoast ? "coasting" : "lost";
    return { pose: this.lastPose, fresh: false, id: this.lockedId, status };
  }
}
