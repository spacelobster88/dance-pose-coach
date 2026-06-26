// Unit tests for the LIVE-WEBCAM multi-person pick & hold UX (#13).
//
// Two concerns, both pure/headless so they run under `node:test` with no DOM:
//
//   1. Coordinate mapping — `mapDisplayToSource` turns a click in CSS/display
//      pixels into source-video pixel space, undoing the `object-fit: contain`
//      letterbox and the horizontal selfie mirror. This is what lets a tap on
//      the mirrored webcam feed land on the right dancer.
//
//   2. Live lock state transitions — exercises the REAL `PoseTracker` +
//      `TargetLock` from src/pose/tracker.ts (the same instances app.ts uses for
//      the live path) with synthetic poses: auto-lock the central dancer, hold
//      that id when a bystander crosses, and tap-relock to a different person.
//
//   node --import ./test/register.mjs --test ./test/liveLock.test.ts
//
// (the npm "test" script chains this after insights + tracker.)

import test from "node:test";
import assert from "node:assert/strict";

import { mapDisplayToSource } from "../src/render/skeleton.ts";
import { PoseTracker, TargetLock } from "../src/pose/tracker.ts";
import type { Pose, Keypoint } from "../src/pose/keypoints.ts";

// ---------------------------------------------------------------------------
// Coordinate mapping
// ---------------------------------------------------------------------------

// A DOMRect-like display box (only the fields the mapper reads).
const rect = (left: number, top: number, width: number, height: number) => ({
  left,
  top,
  width,
  height,
});

test("mapDisplayToSource: center maps to center (no letterbox, no mirror)", () => {
  // Display box exactly matches the video aspect — no letterbox.
  const r = rect(0, 0, 640, 360);
  const p = mapDisplayToSource(320, 180, r, 640, 360, false);
  assert.ok(p);
  assert.ok(Math.abs(p!.x - 320) < 1e-6, `x=${p!.x}`);
  assert.ok(Math.abs(p!.y - 180) < 1e-6, `y=${p!.y}`);
});

test("mapDisplayToSource: corners map to source corners", () => {
  const r = rect(0, 0, 640, 360);
  const tl = mapDisplayToSource(0, 0, r, 640, 360, false);
  const br = mapDisplayToSource(640, 360, r, 640, 360, false);
  assert.ok(tl && br);
  assert.ok(Math.abs(tl!.x - 0) < 1e-6 && Math.abs(tl!.y - 0) < 1e-6);
  assert.ok(Math.abs(br!.x - 640) < 1e-6 && Math.abs(br!.y - 360) < 1e-6);
});

test("mapDisplayToSource: pillarbox (wide display) — center still center, margins rejected", () => {
  // Video 640x360 (16:9) shown in a 800x360 box → letterbox bars left/right.
  // Rendered content width = 360*(640/360)=640, so 80px bars each side.
  const r = rect(0, 0, 800, 360);
  const center = mapDisplayToSource(400, 180, r, 640, 360, false);
  assert.ok(center);
  assert.ok(Math.abs(center!.x - 320) < 1e-6, `x=${center!.x}`);
  assert.ok(Math.abs(center!.y - 180) < 1e-6, `y=${center!.y}`);

  // A click inside the left letterbox bar (x=40 < 80) is outside the video.
  const inBar = mapDisplayToSource(40, 180, r, 640, 360, false);
  assert.equal(inBar, null);

  // Left edge of actual content (x=80) maps to source x≈0.
  const leftEdge = mapDisplayToSource(80, 180, r, 640, 360, false);
  assert.ok(leftEdge);
  assert.ok(Math.abs(leftEdge!.x - 0) < 1e-6, `x=${leftEdge!.x}`);
});

test("mapDisplayToSource: letterbox (tall display) — top/bottom bars rejected", () => {
  // Video 640x360 shown in 640x500 box → content height=360, 70px bars top/bottom.
  const r = rect(0, 0, 640, 500);
  const inTopBar = mapDisplayToSource(320, 30, r, 640, 360, false);
  assert.equal(inTopBar, null);
  const topEdge = mapDisplayToSource(320, 70, r, 640, 360, false);
  assert.ok(topEdge);
  assert.ok(Math.abs(topEdge!.y - 0) < 1e-6, `y=${topEdge!.y}`);
});

test("mapDisplayToSource: mirrored=true flips X about the source center", () => {
  const r = rect(0, 0, 640, 360);
  const plain = mapDisplayToSource(100, 180, r, 640, 360, false);
  const mir = mapDisplayToSource(100, 180, r, 640, 360, true);
  assert.ok(plain && mir);
  // Mirroring flips around the vertical center: x -> W - x. Y unchanged.
  assert.ok(Math.abs(mir!.x - (640 - plain!.x)) < 1e-6, `mir.x=${mir!.x}`);
  assert.ok(Math.abs(mir!.y - plain!.y) < 1e-6);
  // Concretely: display x=100 → source x=100 unmirrored, 540 mirrored.
  assert.ok(Math.abs(plain!.x - 100) < 1e-6);
  assert.ok(Math.abs(mir!.x - 540) < 1e-6);
});

test("mapDisplayToSource: respects a non-zero rect origin (offset on the page)", () => {
  const r = rect(200, 100, 640, 360);
  const p = mapDisplayToSource(200 + 320, 100 + 180, r, 640, 360, false);
  assert.ok(p);
  assert.ok(Math.abs(p!.x - 320) < 1e-6 && Math.abs(p!.y - 180) < 1e-6);
});

// ---------------------------------------------------------------------------
// Live lock state transitions (real tracker + lock)
// ---------------------------------------------------------------------------

const FRAME = { width: 640, height: 360 };

// Build a synthetic pose: a small upright stick figure centered at (cx, cy)
// with high keypoint confidence so poseBBox/scoring treats it as solid.
function personAt(cx: number, cy: number, scale = 60): Pose {
  // A handful of confident keypoints spread around the center so the bbox has
  // real extent (the tracker keys on bbox IoU + keypoint distance).
  const kp = (dx: number, dy: number): Keypoint => ({
    x: cx + dx,
    y: cy + dy,
    score: 0.9,
  });
  const keypoints: Keypoint[] = new Array(17).fill(null).map(() => kp(0, 0));
  keypoints[5] = kp(-scale * 0.4, -scale * 0.5); // left_shoulder
  keypoints[6] = kp(scale * 0.4, -scale * 0.5); // right_shoulder
  keypoints[11] = kp(-scale * 0.3, scale * 0.5); // left_hip
  keypoints[12] = kp(scale * 0.3, scale * 0.5); // right_hip
  keypoints[13] = kp(-scale * 0.3, scale); // left_knee
  keypoints[14] = kp(scale * 0.3, scale); // right_knee
  keypoints[15] = kp(-scale * 0.3, scale * 1.5); // left_ankle
  keypoints[16] = kp(scale * 0.3, scale * 1.5); // right_ankle
  return { keypoints, score: 0.9 };
}

test("live lock: auto-picks the central dancer on the first frame", () => {
  const tracker = new PoseTracker();
  const lock = new TargetLock();
  // A central dancer plus a smaller bystander off to the side.
  const central = personAt(320, 180, 70);
  const bystander = personAt(80, 200, 40);
  const tracked = tracker.update([bystander, central]);
  const sel = lock.select(tracked, { autoPick: true, frame: FRAME });
  assert.equal(sel.status, "tracking");
  assert.ok(sel.fresh);
  // The locked pose should be the central dancer's (largest + most central).
  assert.ok(sel.pose);
  const cx = sel.pose!.keypoints[5].x; // left_shoulder x
  // Central dancer's left_shoulder is near 320-28=292; bystander's near 80-16=64.
  assert.ok(cx > 200, `locked the wrong person (shoulder x=${cx})`);
  assert.notEqual(sel.id, null);
});

test("live lock: holds the same id when a bystander crosses through", () => {
  const tracker = new PoseTracker();
  const lock = new TargetLock();
  // Frame 1: lock onto the central dancer.
  let tracked = tracker.update([personAt(320, 180, 70)]);
  let sel = lock.select(tracked, { autoPick: true, frame: FRAME });
  const lockedId = sel.id;
  assert.notEqual(lockedId, null);

  // Frames 2..N: a bystander walks across while the dancer stays put. The lock
  // must keep following the original dancer, never snapping to the newcomer.
  for (let f = 0; f < 5; f++) {
    const bx = 40 + f * 60;
    tracked = tracker.update([personAt(bx, 200, 45), personAt(320, 180, 70)]);
    sel = lock.select(tracked, { autoPick: true, frame: FRAME });
    assert.equal(sel.status, "tracking", `frame ${f}: status`);
    assert.equal(sel.id, lockedId, `frame ${f}: lock jumped`);
    const cx = sel.pose!.keypoints[5].x;
    assert.ok(cx > 250, `frame ${f}: locked the bystander (x=${cx})`);
  }
});

test("live lock: tap-relock picks the tapped person (via click in source pixels)", () => {
  const tracker = new PoseTracker();
  const lock = new TargetLock();
  // Two dancers; auto-pick takes the central one.
  const left = personAt(140, 200, 55);
  const right = personAt(500, 200, 55);
  let tracked = tracker.update([left, right]);
  let sel = lock.select(tracked, { autoPick: true, frame: FRAME });
  const firstId = sel.id;
  assert.notEqual(firstId, null);

  // Tap on the LEFT dancer (source-pixel click, as the app produces after
  // mapDisplayToSource). The lock should move to the left person.
  tracked = tracker.update([personAt(140, 200, 55), personAt(500, 200, 55)]);
  sel = lock.select(tracked, { click: { x: 140, y: 200 }, frame: FRAME });
  assert.equal(sel.status, "tracking");
  const leftId = sel.id;
  const cx = sel.pose!.keypoints[5].x;
  assert.ok(cx < 250, `tap did not select the left dancer (x=${cx})`);

  // Tap on the RIGHT dancer — lock moves again.
  tracked = tracker.update([personAt(140, 200, 55), personAt(500, 200, 55)]);
  sel = lock.select(tracked, { click: { x: 500, y: 200 }, frame: FRAME });
  const cx2 = sel.pose!.keypoints[5].x;
  assert.ok(cx2 > 350, `tap did not select the right dancer (x=${cx2})`);
  assert.notEqual(sel.id, leftId);
});

test("live lock: pickAt() public relock helper selects the nearest body", () => {
  const tracker = new PoseTracker();
  const lock = new TargetLock();
  const left = personAt(140, 200, 55);
  const right = personAt(500, 200, 55);
  const tracked = tracker.update([left, right]);
  // Direct public relock by point (used by the tap-to-relock handler).
  const picked = lock.pickAt(tracked, { x: 500, y: 200 });
  assert.ok(picked);
  assert.equal(lock.id, picked!.id);
  const cx = picked!.pose.keypoints[5].x;
  assert.ok(cx > 350, `pickAt chose the wrong body (x=${cx})`);
});
