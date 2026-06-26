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
}

interface Track {
  id: number;
  pose: Pose;
  bbox: BBox;
  age: number; // frames since last matched (0 when matched this frame)
  hits: number;
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

  constructor(opts: TrackerOptions = {}) {
    this.iouW = opts.iouWeight ?? 0.6;
    this.distW = opts.distWeight ?? 0.4;
    this.maxCost = opts.maxCost ?? 0.75;
    this.maxAge = opts.maxAge ?? 30;
  }

  reset(): void {
    this.tracks = [];
    this.nextId = 1;
  }

  /** Number of live tracks (including ones currently coasting). */
  get trackCount(): number {
    return this.tracks.length;
  }

  private cost(track: Track, box: BBox, pose: Pose): number {
    const iou = bboxIoU(track.bbox, box);
    let c = this.iouW * (1 - iou);
    const nd = normKeypointDistance(track.pose, pose, track.bbox, box);
    // Saturate the distance term at 1 so a far-away body is ~max cost.
    c += this.distW * Math.min(1, nd ?? 1);
    return c;
  }

  /** Ingest a frame's detections; return them tagged with stable ids. */
  update(poses: Pose[]): TrackedPose[] {
    const boxes = poses.map((p) => poseBBox(p));

    // Candidate (track, detection) pairs within the cost gate.
    const pairs: Array<{ t: number; d: number; c: number }> = [];
    for (let t = 0; t < this.tracks.length; t++) {
      for (let d = 0; d < poses.length; d++) {
        const box = boxes[d];
        if (!box) continue;
        const c = this.cost(this.tracks[t], box, poses[d]);
        if (c <= this.maxCost) pairs.push({ t, d, c });
      }
    }
    pairs.sort((a, b) => a.c - b.c);

    const usedT = new Set<number>();
    const usedD = new Set<number>();
    const out: Array<TrackedPose | undefined> = new Array(poses.length);

    // Greedy: cheapest compatible pairs first.
    for (const { t, d } of pairs) {
      if (usedT.has(t) || usedD.has(d)) continue;
      usedT.add(t);
      usedD.add(d);
      const tr = this.tracks[t];
      const box = boxes[d]!;
      tr.pose = poses[d];
      tr.bbox = box;
      tr.age = 0;
      tr.hits++;
      out[d] = { id: tr.id, pose: poses[d], bbox: box };
    }

    // Age unmatched existing tracks; drop the stale ones. Do this BEFORE adding
    // new tracks so freshly-spawned ids aren't mistaken for unmatched.
    const survivors: Track[] = [];
    for (let t = 0; t < this.tracks.length; t++) {
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
      const tr: Track = { id: this.nextId++, pose: poses[d], bbox: box, age: 0, hits: 1 };
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
  private coast = 0;

  reset(): void {
    this.lockedId = null;
    this.lastPose = null;
    this.lastBBox = null;
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

  private acquire(picked: TrackedPose, status: LockStatus = "tracking"): LockResult {
    this.lockedId = picked.id;
    this.coast = 0;
    this.lastPose = picked.pose;
    this.lastBBox = picked.bbox;
    return { pose: picked.pose, fresh: true, id: picked.id, status };
  }

  /** Resolve which person to follow this frame from the tracked detections. */
  select(tracked: TrackedPose[], opts: SelectOptions = {}): LockResult {
    const reGate = opts.reacquireGate ?? 1.5;
    const maxCoast = opts.maxCoast ?? 30;

    // An explicit click always (re)selects, overriding the current lock.
    if (opts.click) {
      const picked = this.pickByPoint(tracked, opts.click);
      if (picked) return this.acquire(picked);
    }

    // Nothing locked yet: auto-pick a target if asked to.
    if (this.lockedId === null) {
      if (opts.autoPick) {
        const picked = this.autoPick(tracked, opts.frame);
        if (picked) return this.acquire(picked);
      }
      return { pose: null, fresh: false, id: null, status: "searching" };
    }

    // Locked id present this frame → clean live match.
    const match = tracked.find((t) => t.id === this.lockedId);
    if (match) return this.acquire(match);

    // Locked id missing: re-acquire the same body near its last position, but
    // only within the gate — never snap to a far-away person.
    if (this.lastBBox) {
      let best: TrackedPose | null = null;
      let bestD = Infinity;
      const c1 = bboxCenter(this.lastBBox);
      const norm = bboxDiag(this.lastBBox) || 1;
      for (const t of tracked) {
        const c2 = bboxCenter(t.bbox);
        const d = Math.hypot(c1.x - c2.x, c1.y - c2.y) / norm;
        if (d < bestD) {
          bestD = d;
          best = t;
        }
      }
      if (best && bestD <= reGate) return this.acquire(best);
    }

    // No suitable body: coast on the last known pose for a while.
    this.coast++;
    const status: LockStatus = this.coast <= maxCoast ? "coasting" : "lost";
    return { pose: this.lastPose, fresh: false, id: this.lockedId, status };
  }
}
