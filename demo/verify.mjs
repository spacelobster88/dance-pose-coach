// Headless end-to-end verification (no recording).
//
//   node demo/verify.mjs
//
// Spins up vite preview and drives Chrome via Playwright through two scenarios:
//
//   A) File vs file — asserts the live score, per-limb breakdown, and the
//      score-history curve are populated, and that Reset clears the curve.
//   B) Webcam streaming DTW — feeds the test clip to Chrome as a *fake camera*
//      (no real hardware), engages "Live sync", and asserts the lag-compensated
//      score, the score curve, and a real lag readout.
//
// Exits non-zero on any failed assertion.

import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { chromium } from "playwright";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const ASSETS = resolve(__dirname, "assets");
const OUT = resolve(__dirname, "out");
const PORT = 4179;
const BASE = `http://localhost:${PORT}`;
const REF = resolve(ASSETS, "reference.mp4");
const TEST = resolve(ASSETS, "test.mp4");
const CAM_Y4M = resolve(OUT, "cam.y4m");

const log = (m) => console.log(`[verify] ${m}`);
function sh(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (r.status !== 0) throw new Error(`${cmd} ${args.join(" ")} -> ${r.status}`);
}
async function waitForServer(url, timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      if ((await fetch(url)).ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`server ${url} did not start`);
}

const failures = [];
const check = (cond, label, detail) => {
  if (cond) log(`PASS · ${label}${detail ? ` (${detail})` : ""}`);
  else {
    log(`FAIL · ${label}${detail ? ` (${detail})` : ""}`);
    failures.push(label);
  }
};

const CHROME_ARGS = [
  "--use-gl=angle",
  "--use-angle=swiftshader",
  "--enable-unsafe-swiftshader",
  "--ignore-gpu-blocklist",
  "--enable-webgl",
  "--autoplay-policy=no-user-gesture-required",
];

// Count canvas pixels that differ noticeably from the ~#fbfaf8 background.
async function curvePixels(page) {
  return page.evaluate(() => {
    const c = document.getElementById("score-history");
    const ctx = c.getContext("2d");
    const { data } = ctx.getImageData(0, 0, c.width, c.height);
    let nonBg = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
      if (a > 10 && (Math.abs(r - 251) > 24 || Math.abs(g - 250) > 24 || Math.abs(b - 248) > 24)) nonBg++;
    }
    return nonBg;
  });
}

async function gotoReady(page) {
  await page.goto(BASE, { waitUntil: "networkidle" });
  log("waiting for model…");
  await page.waitForFunction(
    () => /ready/i.test(document.getElementById("status")?.textContent || ""),
    null,
    { timeout: 90000 },
  );
}

// ---- Scoring baseline (deterministic, no browser) ----
// Runs the REAL scoring (poseSimilarity from src/pose/similarity.ts) on
// synthetic NormalizedPose objects via `npx tsx`, proving the rebuilt baseline:
//   identical poses score HIGH (>90), a grossly-wrong "arms-down vs arms-up"
//   pair scores LOW (<40), and raising the strictness k LOWERS the score.
// A regression here FAILS `npm run verify`, independent of any browser flake.
const SCORING_SCRIPT = `
import { poseSimilarity, getStrictness } from "../../src/pose/similarity.ts";

// COCO-17 index order (src/pose/keypoints.ts).
const I = {
  left_shoulder: 5, right_shoulder: 6, left_elbow: 7, right_elbow: 8,
  left_wrist: 9, right_wrist: 10, left_hip: 11, right_hip: 12,
  left_knee: 13, right_knee: 14, left_ankle: 15, right_ankle: 16,
};
const p = (x, y) => ({ x, y, valid: true });

// Full 17-point NormalizedPose with arms either DOWN or UP (overhead). Arms-up
// flips the upper-arm/forearm bone directions ~180° — the grossly-wrong case the
// rebuilt baseline must score low. Legs and torso are identical in both poses.
function makePose(armsUp) {
  const pts = Array.from({ length: 17 }, () => ({ x: 0, y: 0, valid: false }));
  pts[I.left_shoulder] = p(-0.4, -1.0);
  pts[I.right_shoulder] = p(0.4, -1.0);
  pts[I.left_hip] = p(-0.3, 0.0);
  pts[I.right_hip] = p(0.3, 0.0);
  pts[I.left_knee] = p(-0.3, 1.0);
  pts[I.right_knee] = p(0.3, 1.0);
  pts[I.left_ankle] = p(-0.3, 2.0);
  pts[I.right_ankle] = p(0.3, 2.0);
  if (armsUp) {
    pts[I.left_elbow] = p(-0.6, -1.8);  pts[I.right_elbow] = p(0.6, -1.8);
    pts[I.left_wrist] = p(-0.7, -2.6);  pts[I.right_wrist] = p(0.7, -2.6);
  } else {
    pts[I.left_elbow] = p(-0.6, -0.2);  pts[I.right_elbow] = p(0.6, -0.2);
    pts[I.left_wrist] = p(-0.7, 0.6);   pts[I.right_wrist] = p(0.7, 0.6);
  }
  return { points: pts };
}

const armsDown = makePose(false);
const armsUp = makePose(true);
const fail = [];
const emit = (cond, label, detail) => {
  console.log(\`SCORING|\${cond ? "PASS" : "FAIL"}|\${label}|\${detail ?? ""}\`);
  if (!cond) fail.push(label);
};

const same = poseSimilarity(armsDown, armsDown);
const sameScore = same ? same.score : NaN;
emit(same !== null && sameScore > 90, "identical pose scores HIGH (>90)", \`score=\${sameScore.toFixed(1)}\`);

const wrong = poseSimilarity(armsDown, armsUp);
const wrongScore = wrong ? wrong.score : NaN;
emit(wrong !== null && wrongScore < 40, "arms-down vs arms-up scores LOW (<40)", \`score=\${wrongScore.toFixed(1)}\`);

emit((sameScore - wrongScore) > 50, "correct vs wrong gap is wide (>50)", \`gap=\${(sameScore - wrongScore).toFixed(1)}\`);

const s6 = (poseSimilarity(armsDown, armsUp, 4, 6) || {}).score;
const s12 = (poseSimilarity(armsDown, armsUp, 4, 12) || {}).score;
emit(s12 < s6, "raising k lowers the score", \`k6=\${s6.toFixed(2)} > k12=\${s12.toFixed(2)}\`);

emit(getStrictness() === 6, "default strictness k is 6", \`k=\${getStrictness()}\`);

process.exit(fail.length ? 1 : 0);
`;

function verifyScoring() {
  log("=== Scoring check: rebuilt baseline (deterministic) ===");
  // Write to demo/out/ so the relative import "../../src/..." resolves.
  const scriptPath = resolve(OUT, "scoring-check.mjs");
  writeFileSync(scriptPath, SCORING_SCRIPT);
  const r = spawnSync("npx", ["tsx", scriptPath], { cwd: ROOT, encoding: "utf8" });
  const out = `${r.stdout || ""}${r.stderr || ""}`;
  for (const line of out.split("\n")) {
    const m = line.match(/^SCORING\|(PASS|FAIL)\|([^|]*)\|(.*)$/);
    if (m) check(m[1] === "PASS", `scoring: ${m[2]}`, m[3] || undefined);
  }
  // If tsx itself blew up (no SCORING lines), surface it as a failure.
  if (!/^SCORING\|/m.test(out)) {
    check(false, "scoring: check script ran", `tsx exit ${r.status}: ${out.trim().slice(0, 300)}`);
  }
}

// ---- 3D / Procrustes viewpoint-invariance check (deterministic, no browser) ----
// Issue B: BlazePose's 3D world landmarks let us cancel the camera angle before
// scoring. Headless swiftshader can't load BlazePose reliably, so instead of
// driving the browser we exercise the REAL 3D math (procrustes.ts + similarity.ts
// + normalize.ts) via `npx tsx` on synthetic poses, mirroring verifyScoring():
//
//   (a) The SAME 3D pose, re-shot from several camera angles (a known rotation
//       applied to its world landmarks), scores ≈100 after Procrustes alignment
//       → viewpoint invariance.
//   (b) A genuinely DIFFERENT 3D pose (arms up vs down) stays LOW even after
//       Procrustes — alignment removes viewpoint, not real posture differences.
//   (c) A 2D-only pose (no points3d) scores EXACTLY as the unchanged 2D path,
//       proving the 3D channel is purely additive.
//
// A regression here FAILS `npm run verify`, independent of any browser flake.
const PROCRUSTES_SCRIPT = `
import { poseSimilarity } from "../../src/pose/similarity.ts";
import { procrustesAlign } from "../../src/pose/procrustes.ts";

// COCO-17 index order (src/pose/keypoints.ts).
const I = {
  left_shoulder: 5, right_shoulder: 6, left_elbow: 7, right_elbow: 8,
  left_wrist: 9, right_wrist: 10, left_hip: 11, right_hip: 12,
  left_knee: 13, right_knee: 14, left_ankle: 15, right_ankle: 16,
};

// Build a full 17-point 3D pose (world landmarks) with arms either DOWN or UP.
// Returns { points3d } in the NormalizedPose shape: a real 3D channel.
function make3d(armsUp) {
  const w = Array.from({ length: 17 }, () => ({ x: 0, y: 0, z: 0, valid: false }));
  const set = (i, x, y, z) => { w[i] = { x, y, z, valid: true }; };
  set(I.left_shoulder, -0.4, -1.0, 0.0);
  set(I.right_shoulder, 0.4, -1.0, 0.0);
  set(I.left_hip, -0.3, 0.0, 0.0);
  set(I.right_hip, 0.3, 0.0, 0.0);
  set(I.left_knee, -0.3, 1.0, 0.0);
  set(I.right_knee, 0.3, 1.0, 0.0);
  set(I.left_ankle, -0.3, 2.0, 0.0);
  set(I.right_ankle, 0.3, 2.0, 0.0);
  if (armsUp) {
    set(I.left_elbow, -0.6, -1.8, 0.1);  set(I.right_elbow, 0.6, -1.8, 0.1);
    set(I.left_wrist, -0.7, -2.6, 0.2);  set(I.right_wrist, 0.7, -2.6, 0.2);
  } else {
    set(I.left_elbow, -0.6, -0.2, 0.1);  set(I.right_elbow, 0.6, -0.2, 0.1);
    set(I.left_wrist, -0.7, 0.6, 0.2);   set(I.right_wrist, 0.7, 0.6, 0.2);
  }
  return w;
}

// 2D points just need to pass the shared-keypoint gate (>=4) in poseSimilarity;
// the 3D path scores off points3d. Project the world points onto x/y so the
// shapes are consistent and the gate is satisfied.
function points2dFrom(w) {
  return w.map((p) => ({ x: p.x, y: p.y, valid: p.valid }));
}

// Rotate world points about the vertical (y) axis by angle a (radians):
// a yaw, i.e. the dancer re-shot from a different horizontal camera angle.
function rotateY(w, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return w.map((p) => p.valid
    ? { x: c * p.x + s * p.z, y: p.y, z: -s * p.x + c * p.z, valid: true }
    : { x: 0, y: 0, z: 0, valid: false });
}
// Compose with a tilt about x so it's a genuine 3D viewpoint change, not planar.
function rotateX(w, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return w.map((p) => p.valid
    ? { x: p.x, y: c * p.y - s * p.z, z: s * p.y + c * p.z, valid: true }
    : { x: 0, y: 0, z: 0, valid: false });
}

const fail = [];
const emit = (cond, label, detail) => {
  console.log(\`PROC|\${cond ? "PASS" : "FAIL"}|\${label}|\${detail ?? ""}\`);
  if (!cond) fail.push(label);
};

const baseWorld = make3d(false);
const refPose = { points: points2dFrom(baseWorld), points3d: baseWorld };

// (a) Viewpoint invariance: same pose re-shot from several camera angles.
//     We rotate the world landmarks (yaw + a tilt) and assert the score is ~100.
const angles = [15, 30, 45, 60, 90, 120];
let minRotScore = Infinity;
for (const deg of angles) {
  const a = (deg * Math.PI) / 180;
  const rotWorld = rotateX(rotateY(baseWorld, a), a * 0.5);
  const testPose = { points: points2dFrom(rotWorld), points3d: rotWorld };
  const r = poseSimilarity(refPose, testPose);
  const score = r ? r.score : NaN;
  minRotScore = Math.min(minRotScore, score);
  emit(r !== null && score > 98, \`rotated \${deg}deg pose still scores HIGH (>98)\`, \`score=\${score.toFixed(2)}\`);
}
emit(minRotScore > 98, "viewpoint invariance across ALL angles (min >98)", \`min=\${minRotScore.toFixed(2)}\`);

// Sanity: the rotation really did move the raw 3D points (so the high score is
// Procrustes earning its keep, not a no-op). Compare a wrist before/after.
{
  const a = (90 * Math.PI) / 180;
  const rot = rotateY(baseWorld, a)[I.left_wrist];
  const orig = baseWorld[I.left_wrist];
  const moved = Math.hypot(rot.x - orig.x, rot.y - orig.y, rot.z - orig.z);
  emit(moved > 0.3, "rotation genuinely moved the raw 3D points", \`wrist delta=\${moved.toFixed(3)}\`);
  // And Procrustes recovers an actual rotation (not identity) for it.
  const aligned = procrustesAlign(baseWorld, rotateY(baseWorld, a), false);
  const offDiag = aligned ? Math.abs(aligned.rotation[0][2]) + Math.abs(aligned.rotation[2][0]) : 0;
  emit(aligned !== null && offDiag > 0.5, "Procrustes recovered a non-identity rotation", \`|R02|+|R20|=\${offDiag.toFixed(3)}\`);
}

// (b) Genuinely different 3D pose (arms up vs down) stays LOW even after align.
{
  const upWorld = make3d(true);
  const upPose = { points: points2dFrom(upWorld), points3d: upWorld };
  const r = poseSimilarity(refPose, upPose);
  const score = r ? r.score : NaN;
  emit(r !== null && score < 50, "different 3D pose (arms up) stays LOW (<50)", \`score=\${score.toFixed(2)}\`);
  // And a rotated copy of the DIFFERENT pose is still low — Procrustes can't
  // fake a match it shouldn't make.
  const a = (40 * Math.PI) / 180;
  const upRot = rotateY(upWorld, a);
  const upRotPose = { points: points2dFrom(upRot), points3d: upRot };
  const r2 = poseSimilarity(refPose, upRotPose);
  const s2 = r2 ? r2.score : NaN;
  emit(r2 !== null && s2 < 50, "different 3D pose stays LOW even when rotated (<50)", \`score=\${s2.toFixed(2)}\`);
}

// (c) 2D-only poses (no points3d) score EXACTLY as the unchanged 2D path.
{
  const ref2d = { points: points2dFrom(baseWorld) };          // no points3d
  const refUp2d = { points: points2dFrom(make3d(true)) };     // no points3d
  const same = poseSimilarity(ref2d, ref2d);
  const diff = poseSimilarity(ref2d, refUp2d);
  emit(same !== null && Math.abs(same.score - 100) < 1e-6, "2D-only identical pose scores 100 (2D path unchanged)", \`score=\${(same ? same.score : NaN).toFixed(6)}\`);
  emit(diff !== null && diff.score < 50, "2D-only different pose scores LOW (2D path unchanged)", \`score=\${(diff ? diff.score : NaN).toFixed(2)}\`);

  // The clincher: feeding the SAME 2D points with vs without a points3d channel
  // must change behavior ONLY when 3D is present. Here, identical 2D points
  // viewed as 2D-only score 100; that exact number is the unchanged 2D path.
  // Cross-check: a 3D pose's 2D-only projection scores the plain 2D number.
  const proj = { points: points2dFrom(baseWorld) };
  const projScore = poseSimilarity(proj, proj).score;
  emit(Math.abs(projScore - 100) < 1e-6, "2D path byte-identical with/without 3D removed", \`score=\${projScore.toFixed(6)}\`);
}

process.exit(fail.length ? 1 : 0);
`;

function verify3D() {
  log("=== 3D / Procrustes check: viewpoint invariance (deterministic) ===");
  const scriptPath = resolve(OUT, "procrustes-check.mjs");
  writeFileSync(scriptPath, PROCRUSTES_SCRIPT);
  const r = spawnSync("npx", ["tsx", scriptPath], { cwd: ROOT, encoding: "utf8" });
  const out = `${r.stdout || ""}${r.stderr || ""}`;
  for (const line of out.split("\n")) {
    const m = line.match(/^PROC\|(PASS|FAIL)\|([^|]*)\|(.*)$/);
    if (m) check(m[1] === "PASS", `3D: ${m[2]}`, m[3] || undefined);
  }
  if (!/^PROC\|/m.test(out)) {
    check(false, "3D: check script ran", `tsx exit ${r.status}: ${out.trim().slice(0, 300)}`);
  }
}

// ---- Sync-calibration estimator (deterministic, no browser) ----
// Issue C (eng-C1): a live camera feed has fixed *machine* latency (transport
// delay) that must be kept separate from the dancer's human *reaction* lag. The
// pure estimator in src/pose/syncCalib.ts turns calibration samples into a
// transport estimate and subtracts it from a measured total lag to leave the
// reaction lag alone. We exercise the REAL module via `npx tsx`, mirroring
// verifyScoring()/verify3D(): transport per-sample = observed-emitted (clamped
// >=0); estimateTransport takes the MEDIAN so a single outlier sample is
// rejected; reactionLag = total-transport (clamped >=0); empty samples are a safe
// {transportMs:0, ok:false}; and the module seam defaults to 0 then persists a set
// value. A regression here FAILS `npm run verify`, independent of any browser flake.
const SYNCCALIB_SCRIPT = `
import {
  transportDelay,
  estimateTransport,
  reactionLag,
  getTransportOffsetMs,
  setTransportOffsetMs,
} from "../../src/pose/syncCalib.ts";

const fail = [];
const emit = (cond, label, detail) => {
  console.log(\`CALIB|\${cond ? "PASS" : "FAIL"}|\${label}|\${detail ?? ""}\`);
  if (!cond) fail.push(label);
};

// transportDelay = observed - emitted (the machine round-trip).
const td = transportDelay(1000, 1150);
emit(td === 150, "transportDelay(1000,1150) === 150", \`got=\${td}\`);
// Negative / clock-skew raw delay clamps to 0 (unphysical transport).
emit(transportDelay(1200, 1000) === 0, "transportDelay clamps negative to 0", \`got=\${transportDelay(1200, 1000)}\`);

// estimateTransport MEDIAN rejects an outlier sample. Four ~150ms samples plus
// one wildly mis-detected clap (delta 5000) -> median stays ~150, NOT skewed.
const samples = [
  { emittedMs: 1000, observedMs: 1150 }, // 150
  { emittedMs: 2000, observedMs: 2140 }, // 140
  { emittedMs: 3000, observedMs: 3160 }, // 160
  { emittedMs: 4000, observedMs: 4155 }, // 155
  { emittedMs: 5000, observedMs: 10000 }, // 5000  <- outlier
];
const est = estimateTransport(samples);
emit(est.ok === true && est.samples === 5, "estimateTransport reports ok + sample count", \`ok=\${est.ok} n=\${est.samples}\`);
emit(est.transportMs >= 140 && est.transportMs <= 160, "estimateTransport median rejects the 5000ms outlier (~150)", \`transportMs=\${est.transportMs}\`);
// Prove it's a median, not a mean: the mean of those deltas is ~1121, far above 160.
const mean = (150 + 140 + 160 + 155 + 5000) / 5;
emit(est.transportMs < mean - 800, "estimate is a median, not a mean (outlier-robust)", \`median=\${est.transportMs} vs mean=\${mean.toFixed(0)}\`);

// reactionLag = total - transport (the dancer alone), clamped >= 0.
const rl = reactionLag(400, 150);
emit(rl === 250, "reactionLag(400,150) === 250 (transport removed)", \`got=\${rl}\`);
emit(reactionLag(100, 150) === 0, "reactionLag clamps negative to 0", \`got=\${reactionLag(100, 150)}\`);

// Empty samples -> safe uncalibrated fallback (backward compatible).
const empty = estimateTransport([]);
emit(empty.transportMs === 0 && empty.ok === false && empty.samples === 0,
  "empty samples -> {transportMs:0, ok:false}", \`transportMs=\${empty.transportMs} ok=\${empty.ok} n=\${empty.samples}\`);

// Module seam: defaults to 0, then persists a set value (and clamps junk to 0).
emit(getTransportOffsetMs() === 0, "transport seam defaults to 0", \`got=\${getTransportOffsetMs()}\`);
setTransportOffsetMs(180);
emit(getTransportOffsetMs() === 180, "setTransportOffsetMs(180) persists", \`got=\${getTransportOffsetMs()}\`);
setTransportOffsetMs(-5);
emit(getTransportOffsetMs() === 0, "setTransportOffsetMs clamps negative to 0", \`got=\${getTransportOffsetMs()}\`);
setTransportOffsetMs(0); // restore default seam state

process.exit(fail.length ? 1 : 0);
`;

function verifySyncCalib() {
  log("=== Sync-calibration estimator: transport vs reaction (deterministic) ===");
  const scriptPath = resolve(OUT, "synccalib-check.mjs");
  writeFileSync(scriptPath, SYNCCALIB_SCRIPT);
  const r = spawnSync("npx", ["tsx", scriptPath], { cwd: ROOT, encoding: "utf8" });
  const out = `${r.stdout || ""}${r.stderr || ""}`;
  for (const line of out.split("\n")) {
    const m = line.match(/^CALIB\|(PASS|FAIL)\|([^|]*)\|(.*)$/);
    if (m) check(m[1] === "PASS", `syncCalib: ${m[2]}`, m[3] || undefined);
  }
  if (!/^CALIB\|/m.test(out)) {
    check(false, "syncCalib: check script ran", `tsx exit ${r.status}: ${out.trim().slice(0, 300)}`);
  }
}

// ---- Streaming aligner: adaptive lag window + transport offset (deterministic) ----
// Issue C (eng-C2): StreamingAligner reports the dancer's *reaction* lag, with the
// calibrated machine transport delay removed, and widens its lag window adaptively
// so a frame that lands beyond the old fixed 700ms cap is still admitted once a
// transport offset is injected. We drive the REAL StreamingAligner via `npx tsx`
// with an injectable offsetSource (the constructor seam), mirroring the other
// deterministic checks:
//   (a) offset 0 -> reported lagMs == totalLagMs (baseline; transport removes nothing).
//   (b) offset 150 -> reported lagMs == totalLagMs - 150 (reaction lag alone).
//   (c) a reference frame ~900ms behind (beyond the old 700ms cap) is NOT matched
//       at offset 0, but IS admitted once an offset widens the adaptive window.
// A regression here FAILS `npm run verify`, independent of any browser flake.
const STREAMING_SCRIPT = `
import { StreamingAligner } from "../../src/pose/streamDtw.ts";

// Minimal NormalizedPose helper: a full 17-point COCO pose. A scalar "phase"
// shifts the arms so distinct phases are genuinely different poses (low score),
// while the same phase round-trips to a HIGH self-similarity — enough for the
// aligner's best-match-in-window logic to lock onto the right reference frame.
const I = {
  left_shoulder: 5, right_shoulder: 6, left_elbow: 7, right_elbow: 8,
  left_wrist: 9, right_wrist: 10, left_hip: 11, right_hip: 12,
  left_knee: 13, right_knee: 14, left_ankle: 15, right_ankle: 16,
};
function pose(phase) {
  const pts = Array.from({ length: 17 }, () => ({ x: 0, y: 0, valid: false }));
  const p = (i, x, y) => { pts[i] = { x, y, valid: true }; };
  p(I.left_shoulder, -0.4, -1.0); p(I.right_shoulder, 0.4, -1.0);
  p(I.left_hip, -0.3, 0.0);       p(I.right_hip, 0.3, 0.0);
  p(I.left_knee, -0.3, 1.0);      p(I.right_knee, 0.3, 1.0);
  p(I.left_ankle, -0.3, 2.0);     p(I.right_ankle, 0.3, 2.0);
  // Arms swing with phase: a clearly different bone direction per phase so the
  // window best-match is unambiguous.
  const dx = Math.cos(phase) * 0.7;
  const dy = Math.sin(phase) * 0.7;
  p(I.left_elbow, -0.5 + dx, -0.5 + dy);  p(I.right_elbow, 0.5 + dx, -0.5 + dy);
  p(I.left_wrist, -0.6 + dx * 1.6, -0.2 + dy * 1.6);
  p(I.right_wrist, 0.6 + dx * 1.6, -0.2 + dy * 1.6);
  return { points: pts };
}

const fail = [];
const emit = (cond, label, detail) => {
  console.log(\`STREAM|\${cond ? "PASS" : "FAIL"}|\${label}|\${detail ?? ""}\`);
  if (!cond) fail.push(label);
};

// A reference timeline of distinct poses, one every 100ms.
const STEP = 100;
const N = 40; // 0..3900ms of reference history
function buildRefs(aligner) {
  for (let k = 0; k < N; k++) aligner.pushRef(pose(k * 0.25), k * STEP);
}

// The live dancer is reproducing the pose from N ms ago: we match the reference
// pose at time (latest - behind) by feeding that exact pose.
const latestT = (N - 1) * STEP; // 3900
const phaseAt = (t) => (t / STEP) * 0.25;

// (a) Baseline: offset 0 -> reaction lag == total lag (transport removes nothing).
{
  const aligner = new StreamingAligner(700, 120, () => 0);
  buildRefs(aligner);
  const behind = 300; // 300ms lag, well within the 700ms base window
  const live = pose(phaseAt(latestT - behind));
  const m = aligner.match(live);
  emit(m !== null, "offset 0: match found within base window", m ? \`score=\${m.score.toFixed(1)}\` : "null");
  emit(m !== null && m.totalLagMs === behind, "offset 0: totalLagMs == injected lag", m ? \`total=\${m.totalLagMs}\` : "null");
  emit(m !== null && m.lagMs === m.totalLagMs, "offset 0: reaction lagMs == total lag (baseline)", m ? \`lagMs=\${m.lagMs} total=\${m.totalLagMs}\` : "null");
  emit(m !== null && m.transportMs === 0, "offset 0: transportMs == 0", m ? \`transport=\${m.transportMs}\` : "null");
}

// (b) Injected offset 150: reaction lag == total - 150 (transport removed).
{
  const OFF = 150;
  const aligner = new StreamingAligner(700, 120, () => OFF);
  buildRefs(aligner);
  const behind = 400;
  const live = pose(phaseAt(latestT - behind));
  const m = aligner.match(live);
  emit(m !== null && m.totalLagMs === behind, "offset 150: totalLagMs == injected lag", m ? \`total=\${m.totalLagMs}\` : "null");
  emit(m !== null && m.transportMs === OFF, "offset 150: transportMs == injected offset", m ? \`transport=\${m.transportMs}\` : "null");
  emit(m !== null && m.lagMs === behind - OFF, "offset 150: reaction lagMs == total - 150", m ? \`lagMs=\${m.lagMs} (expected \${behind - OFF})\` : "null");
}

// (c) Adaptive window: a reference frame ~900ms behind sits BEYOND the old fixed
// 700ms cap. At offset 0 it must be rejected; once an offset widens the window
// (700 + offset + 250 headroom) the same frame is admitted.
{
  const behind = 900; // beyond the 700ms base cap
  const livePhase = phaseAt(latestT - behind);

  const baseline = new StreamingAligner(700, 120, () => 0);
  buildRefs(baseline);
  const mNarrow = baseline.match(pose(livePhase));
  // With a 700ms window the best match cannot be the 900ms-old frame; the matched
  // total lag must stay <= 700 (the old frame is simply not in scan range).
  emit(mNarrow !== null && mNarrow.totalLagMs <= 700,
    "offset 0: 900ms-old frame is OUTSIDE the 700ms window (not matched)",
    mNarrow ? \`total=\${mNarrow.totalLagMs}\` : "null");

  const widened = new StreamingAligner(700, 120, () => 600);
  buildRefs(widened);
  const mWide = widened.match(pose(livePhase));
  // 700 + 600 + 250 = 1550ms window -> the 900ms-old frame is now in scan range
  // and is the best match, so the reported total lag reaches ~900ms.
  emit(mWide !== null && mWide.totalLagMs === behind,
    "widened window admits the 900ms-old frame (adaptive lag)",
    mWide ? \`total=\${mWide.totalLagMs}\` : "null");
  // And the reaction lag has the transport (600) removed: 900 - 600 = 300.
  emit(mWide !== null && mWide.lagMs === behind - 600,
    "widened: reaction lagMs == total - transport", mWide ? \`lagMs=\${mWide.lagMs} (expected \${behind - 600})\` : "null");
}

process.exit(fail.length ? 1 : 0);
`;

function verifyStreaming() {
  log("=== Streaming aligner: adaptive lag + transport offset (deterministic) ===");
  const scriptPath = resolve(OUT, "streaming-check.mjs");
  writeFileSync(scriptPath, STREAMING_SCRIPT);
  const r = spawnSync("npx", ["tsx", scriptPath], { cwd: ROOT, encoding: "utf8" });
  const out = `${r.stdout || ""}${r.stderr || ""}`;
  for (const line of out.split("\n")) {
    const m = line.match(/^STREAM\|(PASS|FAIL)\|([^|]*)\|(.*)$/);
    if (m) check(m[1] === "PASS", `streaming: ${m[2]}`, m[3] || undefined);
  }
  if (!/^STREAM\|/m.test(out)) {
    check(false, "streaming: check script ran", `tsx exit ${r.status}: ${out.trim().slice(0, 300)}`);
  }
}

// ---- Bundle cleanliness: model weights load from CDN, not bundled ----
// Issue B requires BlazePose weights to stream from cdn.jsdelivr.net at runtime,
// so the built bundle must contain NO model-weight files (*.tflite, *.binarypb,
// *.wasm, *.data, *pose* binaries) yet MUST embed the CDN solutionPath string.
function verifyBundleCleanliness() {
  log("=== Bundle cleanliness: weights from CDN, not bundled ===");
  const dist = resolve(ROOT, "dist");
  // Recursively list dist and flag any model-weight artifacts.
  const weightFiles = [];
  const walk = (dir) => {
    for (const ent of readdirSync(dir, { withFileTypes: true })) {
      const full = resolve(dir, ent.name);
      if (ent.isDirectory()) { walk(full); continue; }
      const lower = ent.name.toLowerCase();
      if (
        /\.(tflite|binarypb|wasm|data)$/.test(lower) ||
        (/pose/.test(lower) && /\.(bin|tflite|binarypb|wasm|data|task)$/.test(lower))
      ) {
        weightFiles.push(full.replace(`${dist}/`, ""));
      }
    }
  };
  walk(dist);
  check(weightFiles.length === 0, "dist contains NO model-weight files",
    weightFiles.length ? weightFiles.join(", ") : "none found");

  // The CDN solutionPath string must be present in the built JS so BlazePose can
  // stream its weights at runtime.
  const CDN = "cdn.jsdelivr.net/npm/@mediapipe/pose";
  let found = false;
  const grep = (dir) => {
    for (const ent of readdirSync(dir, { withFileTypes: true })) {
      const full = resolve(dir, ent.name);
      if (ent.isDirectory()) { grep(full); continue; }
      if (/\.(js|mjs|html)$/.test(ent.name)) {
        if (readFileSync(full, "utf8").includes(CDN)) { found = true; return; }
      }
    }
  };
  grep(dist);
  check(found, `CDN URL string present in built JS (${CDN})`, found ? "present" : "MISSING");
}

// ---- Scenario A: file vs file ----
async function verifyFileMode() {
  log("=== Scenario A: file vs file ===");
  const browser = await chromium.launch({ channel: "chrome", headless: true, args: CHROME_ARGS });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    page.on("console", (m) => m.type() === "error" && log(`page error: ${m.text()}`));
    await gotoReady(page);
    await page.setInputFiles("#ref-file", REF);
    await page.setInputFiles("#test-file", TEST);
    await page.waitForFunction(
      () => !document.getElementById("play-btn")?.hasAttribute("disabled"),
      null,
      { timeout: 30000 },
    );
    log("playing…");
    await page.click("#play-btn");
    await page.waitForFunction(
      () => {
        const t = document.getElementById("score-now")?.textContent || "—";
        return t !== "—" && Number(t) > 0;
      },
      null,
      { timeout: 30000 },
    );
    await page.waitForTimeout(4000);

    const scoreNow = await page.$eval("#score-now", (e) => e.textContent);
    check(Number(scoreNow) > 0, "A: live similarity score present", `now=${scoreNow}`);

    const limbVals = await page.$$eval(".bd-val", (els) => els.map((e) => e.textContent));
    const realLimbs = limbVals.filter((v) => v && v !== "—").length;
    check(realLimbs >= 3, "A: per-limb bars populated", `${realLimbs} real values`);

    const nonBg = await curvePixels(page);
    check(nonBg > 200, "A: score-history curve drawn", `${nonBg} non-bg px`);

    // Strictness slider (uiux-1 / eng-A2): the control exists in the DOM and is
    // a 2..14 range defaulting to 6, wired to live re-scoring.
    const strict = await page.evaluate(() => {
      const el = document.getElementById("strictness");
      if (!el) return null;
      return { type: el.type, min: el.min, max: el.max, value: el.value };
    });
    check(
      strict !== null && strict.type === "range" &&
        Number(strict.min) === 2 && Number(strict.max) === 14,
      "A: strictness slider #strictness present (2..14 range)",
      strict ? `default=${strict.value}` : "missing",
    );

    // Raising k (strictness) must LOWER the live score on the same frame.
    // Pause first so the slider re-scores a single frozen pose pair
    // (app.ts rescoreCurrentFrame) rather than racing live playback.
    if (!(await page.$eval("#play-btn", (e) => /play/i.test(e.textContent)))) {
      await page.click("#play-btn"); // toggle to Pause
      await page.waitForTimeout(300);
    }
    const beforeK = Number(await page.$eval("#score-now", (e) => e.textContent));
    if (strict) {
      // Dispatch only "input": app.ts re-scores the frozen frame live on input,
      // while "change" intentionally restarts the score history (resets to "—").
      await page.$eval("#strictness", (el) => {
        el.value = "14";
        el.dispatchEvent(new Event("input", { bubbles: true }));
      });
      await page.waitForTimeout(400);
      const afterK = Number(await page.$eval("#score-now", (e) => e.textContent));
      check(
        Number.isFinite(beforeK) && Number.isFinite(afterK) && afterK < beforeK,
        "A: raising strictness lowers the live score",
        `k6=${beforeK} -> k14=${afterK}`,
      );
      // Restore default so downstream state is unchanged.
      await page.$eval("#strictness", (el) => {
        el.value = "6";
        el.dispatchEvent(new Event("input", { bubbles: true }));
      });
    }

    // AI coaching (#15): with frames recorded from the run, the offline
    // rule-based provider must turn the report into rendered Markdown coaching.
    // Force the deterministic, no-network provider so this stays hermetic.
    // Drive via evaluate (set value + fire change, then click) so the assertion
    // doesn't depend on the panel being scrolled into view during playback.
    await page.evaluate(() => {
      const sel = document.getElementById("coach-provider");
      sel.value = "rule-based";
      sel.dispatchEvent(new Event("change", { bubbles: true }));
      document.getElementById("coach-btn").click();
    });
    await page.waitForFunction(
      () => {
        const p = document.getElementById("coach-panel");
        return p && !p.classList.contains("empty") && /Analysing/.test(p.textContent) === false &&
          p.querySelector("h3, ul, ol, strong");
      },
      { timeout: 5000 },
    ).catch(() => {});
    const coach = await page.evaluate(() => {
      const p = document.getElementById("coach-panel");
      const s = document.getElementById("coach-source");
      return {
        hasHeading: !!p?.querySelector("h3"),
        text: p?.textContent || "",
        source: s?.textContent || "",
        sourceHidden: s?.hidden ?? true,
      };
    });
    check(coach.hasHeading && coach.text.length > 40, "A: AI coaching panel renders Markdown",
      `${coach.text.length} chars`);
    check(/rule-based/i.test(coach.source) && !coach.sourceHidden,
      "A: coaching source credits the rule-based provider", `"${coach.source}"`);

    await page.click("#restart-btn");
    await page.waitForTimeout(300);
    const afterReset = await page.evaluate(() => {
      const c = document.getElementById("score-history");
      const ctx = c.getContext("2d");
      const { data } = ctx.getImageData(0, 0, c.width, c.height);
      let colored = 0;
      for (let i = 0; i < data.length; i += 4) {
        const max = Math.max(data[i], data[i + 1], data[i + 2]);
        const min = Math.min(data[i], data[i + 1], data[i + 2]);
        if (max - min > 60) colored++;
      }
      return colored;
    });
    check(afterReset < 50, "A: reset clears the curve", `${afterReset} colored px left`);
  } finally {
    await browser.close().catch(() => {});
  }
}

// ---- Scenario B: webcam streaming DTW (fake camera) ----
async function verifyWebcamStreaming() {
  log("=== Scenario B: webcam streaming DTW ===");
  if (!existsSync(CAM_Y4M)) {
    log("building fake-camera y4m from the test clip…");
    sh("ffmpeg", [
      "-hide_banner", "-y", "-i", TEST,
      "-t", "8", "-vf", "scale=320:240,fps=24", "-pix_fmt", "yuv420p", CAM_Y4M,
    ]);
  }
  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: [
      ...CHROME_ARGS,
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream", // auto-grant camera permission
      `--use-file-for-fake-video-capture=${CAM_Y4M}`,
    ],
  });
  try {
    const ctx = await browser.newContext({
      viewport: { width: 1280, height: 900 },
      permissions: ["camera"],
    });
    const page = await ctx.newPage();
    page.on("console", (m) => m.type() === "error" && log(`page error: ${m.text()}`));
    await gotoReady(page);

    // Reference is a file; the "attempt" comes from the fake camera.
    await page.setInputFiles("#ref-file", REF);
    await page.selectOption("#test-source", "webcam");
    await page.waitForFunction(
      () => /webcam live|live\./i.test(document.getElementById("status")?.textContent || ""),
      null,
      { timeout: 30000 },
    );
    await page.waitForFunction(
      () => !document.getElementById("play-btn")?.hasAttribute("disabled"),
      null,
      { timeout: 30000 },
    );

    // The alignment button should now be the live-sync toggle.
    const btnText = await page.$eval("#dtw-btn", (e) => e.textContent.trim());
    check(/live sync/i.test(btnText), "B: button is Live sync in webcam mode", `"${btnText}"`);

    // Issue C: the sync-calibration control appears in webcam mode. The button
    // lives inside the webcam-only #livesync-cluster; assert it's present AND
    // actually visible (not display:none / hidden) now that we're in webcam mode.
    const calib = await page.evaluate(() => {
      const btn = document.getElementById("calib-btn");
      if (!btn) return null;
      const r = btn.getBoundingClientRect();
      return {
        text: btn.textContent.trim(),
        visible: r.width > 0 && r.height > 0 && !btn.hidden,
      };
    });
    check(calib !== null, "B: #calib-btn present in webcam mode", calib ? `"${calib.text}"` : "MISSING");
    check(calib !== null && calib.visible, "B: #calib-btn visible in webcam mode",
      calib ? `visible=${calib.visible}` : "missing");

    // Engage live sync, then play.
    await page.click("#dtw-btn");
    const onText = await page.$eval("#dtw-btn", (e) => e.textContent.trim());
    check(/on/i.test(onText), "B: live sync engaged", `"${onText}"`);
    const lagVisible = await page.$eval("#lag-stat", (e) => !e.hidden);
    check(lagVisible, "B: lag stat visible when synced");

    log("playing webcam follow…");
    await page.click("#play-btn");
    await page.waitForFunction(
      () => {
        const t = document.getElementById("score-now")?.textContent || "—";
        return t !== "—" && Number(t) > 0;
      },
      null,
      { timeout: 30000 },
    );
    await page.waitForTimeout(4000);

    const scoreNow = await page.$eval("#score-now", (e) => e.textContent);
    check(Number(scoreNow) > 0, "B: lag-compensated score present", `now=${scoreNow}`);

    const lagText = await page.$eval("#score-lag", (e) => e.textContent);
    check(lagText !== "—" && Number.isFinite(Number(lagText)) && Number(lagText) >= 0,
      "B: real lag readout", `lag=${lagText}ms`);

    const nonBg = await curvePixels(page);
    check(nonBg > 200, "B: score-history curve drawn (webcam)", `${nonBg} non-bg px`);
  } finally {
    await browser.close().catch(() => {});
  }
}

// ---- Scenario C: export a scored comparison clip ----
async function verifyExport() {
  log("=== Scenario C: export scored clip ===");
  const browser = await chromium.launch({ channel: "chrome", headless: true, args: CHROME_ARGS });
  try {
    const ctx = await browser.newContext({
      viewport: { width: 1280, height: 900 },
      acceptDownloads: true,
    });
    const page = await ctx.newPage();
    page.on("console", (m) => m.type() === "error" && log(`page error: ${m.text()}`));
    await gotoReady(page);
    await page.setInputFiles("#ref-file", REF);
    await page.setInputFiles("#test-file", TEST);
    await page.waitForFunction(
      () => !document.getElementById("play-btn")?.hasAttribute("disabled"),
      null,
      { timeout: 30000 },
    );

    const downloadPromise = page.waitForEvent("download", { timeout: 30000 });
    log("recording…");
    await page.click("#record-btn"); // starts recording + playback
    const recText = await page.$eval("#record-btn", (e) => e.textContent.trim());
    check(/stop/i.test(recText), "C: record button shows Stop while recording", `"${recText}"`);

    await page.waitForTimeout(5000); // capture a few seconds of frames
    await page.click("#record-btn"); // stop -> triggers download

    const download = await downloadPromise;
    const name = download.suggestedFilename();
    const saved = resolve(OUT, name);
    await download.saveAs(saved);
    const size = statSync(saved).size;
    // recorder.ts (ExportFormat) prefers MP4 when the browser supports it and
    // falls back to WebM, so the exported clip is legitimately either container.
    check(/\.(webm|mp4)$/.test(name), "C: download is a .webm or .mp4", name);
    check(size > 20000, "C: recorded clip is non-empty", `${(size / 1024).toFixed(0)} KB`);

    // Decode it: a valid clip reports the composite resolution. (MediaRecorder
    // output — webm, and fragmented mp4 — may omit a container duration, so we
    // count decoded frames instead of trusting a duration field.)
    const dim = spawnSync("ffprobe", [
      "-v", "error", "-select_streams", "v:0",
      "-show_entries", "stream=width,height",
      "-of", "default=nw=1:nk=1", saved,
    ], { encoding: "utf8" });
    const [w, h] = (dim.stdout || "").trim().split(/\s+/).map(Number);
    check(w === 1280 && h === 596, "C: clip decodes at composite resolution", `${w}x${h}`);

    const frames = spawnSync("ffprobe", [
      "-v", "error", "-select_streams", "v:0",
      "-count_packets", "-show_entries", "stream=nb_read_packets",
      "-of", "default=nw=1:nk=1", saved,
    ], { encoding: "utf8" });
    const nFrames = Number((frames.stdout || "").trim());
    check(nFrames > 20, "C: clip contains real frames", `${nFrames} frames`);
  } finally {
    await browser.close().catch(() => {});
  }
}

async function main() {
  if (!existsSync(REF) || !existsSync(TEST)) {
    sh("node", [resolve(__dirname, "prepare-assets.mjs")], { cwd: ROOT });
  }
  if (!existsSync(resolve(ROOT, "dist/index.html"))) {
    sh("npm", ["run", "build"], { cwd: ROOT });
  }
  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(OUT, { recursive: true });

  const server = spawn(
    "npx", ["vite", "preview", "--port", String(PORT), "--strictPort"],
    { cwd: ROOT, stdio: "ignore" },
  );
  try {
    await waitForServer(BASE);
    verifyScoring();
    verify3D();
    verifySyncCalib();
    verifyStreaming();
    verifyBundleCleanliness();
    await verifyFileMode();
    await verifyWebcamStreaming();
    await verifyExport();
  } finally {
    server.kill("SIGTERM");
  }

  if (failures.length) {
    console.error(`\n[verify] FAILED: ${failures.join(", ")}`);
    process.exit(1);
  }
  console.log("\n[verify] ALL CHECKS PASSED");
}

main().catch((err) => {
  console.error("[verify] ERROR:", err);
  process.exit(1);
});
