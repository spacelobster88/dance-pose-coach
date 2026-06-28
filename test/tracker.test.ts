// Identity-hardening tests (#12): HSV torso-color appearance cue + Hungarian
// optimal assignment + ambiguity-guarded re-acquire keep the LOCKED dancer's id
// stable through dense crossings and brief occlusion.
//
//   npm test
//
// Plain node:test. The appearance helpers (rgbToHsv / buildHistogram /
// histogramDistance) are pure and operate on number arrays, so the adversarial
// crossing scenario is fully deterministic and headless.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  PoseTracker,
  TargetLock,
  rgbToHsv,
  buildHistogram,
  histogramDistance,
  HIST_BINS,
  type BBox,
  type TrackedPose,
} from "../src/pose/tracker.ts";
import { KEYPOINT_NAMES, type Pose } from "../src/pose/keypoints.ts";

// ---- helpers ----------------------------------------------------------------

// Build a full 17-keypoint pose centered at (cx, cy) with a body of `size`,
// every keypoint confident. We only need a torso + limbs roughly arranged so
// poseBBox produces a sensible box; exact anatomy is irrelevant to identity.
function makePose(cx: number, cy: number, size = 60, appearance?: number[]): Pose {
  const half = size / 2;
  const kp = (dx: number, dy: number) => ({ x: cx + dx, y: cy + dy, score: 0.95 });
  // Map by name so we don't depend on exact index math.
  const offsets: Record<string, [number, number]> = {
    nose: [0, -half],
    left_eye: [-half * 0.1, -half * 1.05],
    right_eye: [half * 0.1, -half * 1.05],
    left_ear: [-half * 0.2, -half],
    right_ear: [half * 0.2, -half],
    left_shoulder: [-half * 0.5, -half * 0.3],
    right_shoulder: [half * 0.5, -half * 0.3],
    left_elbow: [-half * 0.7, 0],
    right_elbow: [half * 0.7, 0],
    left_wrist: [-half * 0.8, half * 0.3],
    right_wrist: [half * 0.8, half * 0.3],
    left_hip: [-half * 0.35, half * 0.3],
    right_hip: [half * 0.35, half * 0.3],
    left_knee: [-half * 0.4, half * 0.65],
    right_knee: [half * 0.4, half * 0.65],
    left_ankle: [-half * 0.4, half],
    right_ankle: [half * 0.4, half],
  };
  const keypoints = KEYPOINT_NAMES.map((name) => {
    const [dx, dy] = offsets[name];
    return { ...kp(dx, dy), name };
  });
  const pose: Pose = { keypoints, score: 0.95 };
  if (appearance) (pose as Pose & { appearance?: number[] }).appearance = appearance;
  return pose;
}

// A solid-color torso histogram: paint the whole torso region one RGB color and
// build the same histogram the detector would. This yields distinct, stable
// fingerprints for "red"/"green"/"blue" dancers.
function solidColorHist(r: number, g: number, b: number): number[] {
  const px: number[] = [];
  for (let i = 0; i < 200; i++) px.push(r, g, b);
  return buildHistogram(px);
}

const RED = solidColorHist(220, 30, 30);
const GREEN = solidColorHist(30, 200, 30);
const BLUE = solidColorHist(30, 30, 220);

// ---- unit: HSV conversion ---------------------------------------------------

test("rgbToHsv: primary colors map to expected hue sextants", () => {
  const [hr] = rgbToHsv(255, 0, 0);
  const [hg] = rgbToHsv(0, 255, 0);
  const [hb] = rgbToHsv(0, 0, 255);
  // Hue in [0,1): red ~0, green ~1/3, blue ~2/3.
  assert.ok(Math.abs(hr - 0) < 0.02 || Math.abs(hr - 1) < 0.02, `red hue ${hr}`);
  assert.ok(Math.abs(hg - 1 / 3) < 0.02, `green hue ${hg}`);
  assert.ok(Math.abs(hb - 2 / 3) < 0.02, `blue hue ${hb}`);

  const [, sGray, vGray] = rgbToHsv(128, 128, 128);
  assert.ok(sGray < 0.01, `gray saturation ${sGray}`);
  assert.ok(Math.abs(vGray - 128 / 255) < 0.01, `gray value ${vGray}`);

  const [, , vBlack] = rgbToHsv(0, 0, 0);
  assert.ok(vBlack < 0.01, `black value ${vBlack}`);
});

// ---- unit: histogram + distance ---------------------------------------------

test("buildHistogram: normalized (sums to ~1) and correct length", () => {
  assert.equal(RED.length, HIST_BINS);
  const sum = RED.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-6, `sum ${sum}`);
});

test("buildHistogram: gates out near-black / near-white / low-sat pixels", () => {
  // All-black and all-white inputs have no chromatic content → empty hist (all 0).
  const black = buildHistogram([0, 0, 0, 0, 0, 0]);
  const white = buildHistogram([255, 255, 255, 255, 255, 255]);
  assert.equal(black.reduce((a, b) => a + b, 0), 0);
  assert.equal(white.reduce((a, b) => a + b, 0), 0);
});

test("histogramDistance: identical→~0, different→high, in [0,1], symmetric", () => {
  assert.ok(histogramDistance(RED, RED) < 1e-6, "identical");
  assert.ok(histogramDistance(RED, GREEN) > 0.7, `red vs green ${histogramDistance(RED, GREEN)}`);
  assert.ok(histogramDistance(RED, BLUE) > 0.7, `red vs blue ${histogramDistance(RED, BLUE)}`);
  const d1 = histogramDistance(RED, GREEN);
  const d2 = histogramDistance(GREEN, RED);
  assert.ok(Math.abs(d1 - d2) < 1e-9, "symmetric");
  assert.ok(d1 >= 0 && d1 <= 1, "bounded");
  // Empty vs empty defined as max distance (1) — no info to match on.
  assert.equal(histogramDistance([], []), 1);
});

// ---- scenario: adversarial crossing -----------------------------------------

// Three dancers in a row. The RED target sweeps left→right and passes THROUGH
// the green and blue dancers (their boxes overlap heavily at crossing frames),
// so geometry alone is ambiguous. Appearance (color) must hold the lock.
function crossingFrame(t: number): Pose[] {
  // t in [0,1]; target moves 40→460, others fixed at 180 and 320.
  const targetX = 40 + 420 * t;
  return [
    makePose(targetX, 200, 70, RED),
    makePose(180, 200, 70, GREEN),
    makePose(320, 200, 70, BLUE),
  ];
}

test("crossing: hardened tracker+lock keeps the SAME id across the crossing", () => {
  const tracker = new PoseTracker();
  const lock = new TargetLock();

  // Frame 0: lock onto the RED dancer via a click at its position.
  let frame = crossingFrame(0);
  let tracked = tracker.update(frame);
  let sel = lock.select(tracked, { click: { x: 40, y: 200 } });
  const lockedId = sel.id;
  assert.notEqual(lockedId, null, "acquired a lock");

  const STEPS = 24;
  let switches = 0;
  let lastId = lockedId;
  for (let i = 1; i <= STEPS; i++) {
    frame = crossingFrame(i / STEPS);
    tracked = tracker.update(frame);
    sel = lock.select(tracked, {});
    // The displayed/scored pose, when fresh, must be the RED dancer.
    if (sel.fresh && sel.pose) {
      const app = (sel.pose as Pose & { appearance?: number[] }).appearance;
      assert.ok(app, `frame ${i}: locked pose carries appearance`);
      const dRed = histogramDistance(app!, RED);
      assert.ok(dRed < 0.2, `frame ${i}: locked pose is RED (dRed=${dRed})`);
    }
    if (sel.id !== lastId) switches++;
    lastId = sel.id;
  }
  assert.equal(switches, 0, `id stayed stable across crossing (switches=${switches})`);
  assert.equal(lock.id, lockedId, "final locked id unchanged");
});

test("crossing is genuinely adversarial: motion-only SWAPS, appearance does NOT", () => {
  // Two near-coincident bodies (perfect geometric tie, no velocity history yet)
  // swap positions AND detection order frame-to-frame. Geometry alone cannot
  // disambiguate — only the color histogram can.
  function xOf(t: { pose: Pose }): number {
    let mn = Infinity,
      mx = -Infinity;
    for (const k of t.pose.keypoints) {
      mn = Math.min(mn, k.x);
      mx = Math.max(mx, k.x);
    }
    return (mn + mx) / 2;
  }
  function run(withColor: boolean): boolean {
    const tr = new PoseTracker({
      appearanceWeight: withColor ? 0.7 : 0,
      motionWeight: withColor ? 0.3 : 1,
      gate: 5,
    });
    let tk = tr.update([
      makePose(200, 200, 70, withColor ? RED : undefined),
      makePose(205, 200, 70, withColor ? GREEN : undefined),
    ]);
    const redId = tk[0].id;
    // Swap both positions and array order; the RED body is now at x≈205.
    tk = tr.update([
      makePose(200, 200, 70, withColor ? GREEN : undefined),
      makePose(205, 200, 70, withColor ? RED : undefined),
    ]);
    const redBody = tk.find((t) => Math.abs(xOf(t) - 205) < 3)!;
    return redBody.id === redId; // true ⇒ id stayed glued to the red body
  }
  assert.equal(run(false), false, "motion-only baseline SWAPS ids (scenario is adversarial)");
  assert.equal(run(true), true, "appearance keeps the id glued to the red body");
});

// ---- scenario: occlusion → re-acquire same person ---------------------------

test("occlusion: target disappears then reappears → re-acquires SAME id", () => {
  const tracker = new PoseTracker();
  const lock = new TargetLock();

  // Establish the RED target plus a GREEN neighbor.
  let tracked = tracker.update([makePose(100, 200, 70, RED), makePose(260, 200, 70, GREEN)]);
  let sel = lock.select(tracked, { click: { x: 100, y: 200 } });
  const lockedId = sel.id;
  assert.notEqual(lockedId, null);

  // 4 frames where RED is occluded (only GREEN visible nearby). Should coast,
  // and must NOT hop onto GREEN.
  for (let i = 0; i < 4; i++) {
    tracked = tracker.update([makePose(255 + i, 200, 70, GREEN)]);
    sel = lock.select(tracked, {});
    assert.notEqual(sel.fresh && sel.id !== lockedId, true, `frame ${i}: did not hop to neighbor`);
  }

  // RED reappears near its last position with the SAME color.
  tracked = tracker.update([makePose(108, 200, 70, RED), makePose(259, 200, 70, GREEN)]);
  sel = lock.select(tracked, {});
  assert.equal(sel.id, lockedId, "re-acquired the same id");
  assert.equal(sel.fresh, true, "fresh match on reappear");
  const app = (sel.pose as Pose & { appearance?: number[] }).appearance;
  assert.ok(app && histogramDistance(app, RED) < 0.2, "re-acquired pose is RED");
});

// ---- scenario: ambiguity guard ----------------------------------------------

// For the ambiguity guard we exercise TargetLock.select directly with a
// hand-built tracked list whose ids do NOT include the locked id — i.e. the
// genuine "locked id is gone" re-acquire path the guard protects. (Round-
// tripping through the tracker would let Hungarian reassign the locked id to
// one of the rivals, which is exactly the drift the tracker prevents and so
// would never reach the lock's guard.)
function trackedPose(id: number, cx: number, cy: number, appearance: number[]): TrackedPose {
  const pose = makePose(cx, cy, 70, appearance);
  const box = bboxOf(pose);
  return { id, pose, bbox: box };
}
function bboxOf(pose: Pose): BBox {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const k of pose.keypoints) {
    minX = Math.min(minX, k.x);
    minY = Math.min(minY, k.y);
    maxX = Math.max(maxX, k.x);
    maxY = Math.max(maxY, k.y);
  }
  return { minX, minY, maxX, maxY };
}

test("ambiguity guard: two same-color look-alikes in the zone → status 'lost'", () => {
  const lock = new TargetLock();

  // Lock RED (id 1).
  let sel = lock.select([trackedPose(1, 200, 200, RED)], { click: { x: 200, y: 200 } });
  const lockedId = sel.id;
  assert.equal(lockedId, 1);

  // Locked id (1) is gone; two indistinguishable RED look-alikes (ids 2 & 3)
  // occupy the predicted zone, both passing motion+appearance gates. The guard
  // must refuse to hop → "lost".
  sel = lock.select([trackedPose(2, 205, 200, RED), trackedPose(3, 195, 215, RED)], {});
  assert.equal(sel.status, "lost", "ambiguous same-color pair → lost, no hop");
  assert.notEqual(sel.fresh, true, "did not emit a fresh wrong-body match");
});

test("ambiguity control: distinct 2nd color → DOES re-acquire", () => {
  const lock = new TargetLock();

  let sel = lock.select([trackedPose(1, 200, 200, RED)], { click: { x: 200, y: 200 } });
  assert.equal(sel.id, 1);

  // Locked id gone; one RED look-alike (id 2) + one clearly-BLUE distractor
  // (id 3) in the zone. RED is unambiguously closer in appearance → re-acquire.
  sel = lock.select([trackedPose(2, 205, 200, RED), trackedPose(3, 195, 215, BLUE)], {});
  assert.notEqual(sel.status, "lost", "unambiguous → re-acquires");
  assert.equal(sel.fresh, true, "fresh match");
  assert.equal(sel.id, 2, "re-acquired the RED look-alike");
  const app = (sel.pose as Pose & { appearance?: number[] }).appearance;
  assert.ok(app && histogramDistance(app, RED) < 0.2, "re-acquired the RED look-alike");
});
