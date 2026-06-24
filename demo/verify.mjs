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
import { existsSync, mkdirSync, rmSync, statSync } from "node:fs";
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
    check(/\.webm$/.test(name), "C: download is a .webm", name);
    check(size > 20000, "C: recorded clip is non-empty", `${(size / 1024).toFixed(0)} KB`);

    // Decode it: a valid clip reports the composite resolution. (MediaRecorder
    // webm omits a container duration, so we count decoded frames instead.)
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
