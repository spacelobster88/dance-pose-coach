// Headless end-to-end verification (no recording).
//
//   node demo/verify.mjs
//
// Spins up vite preview, drives Chrome via Playwright through a full run, and
// asserts the live feature surface is actually populated:
//   - similarity score is a real number,
//   - per-limb breakdown bars carry real values,
//   - the score-history canvas has a drawn curve (non-background pixels),
//   - (optional) DTW toggle engages.
// Exits non-zero on any failed assertion.

import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { existsSync, mkdirSync, rmSync } from "node:fs";
import { chromium } from "playwright";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const ASSETS = resolve(__dirname, "assets");
const OUT = resolve(__dirname, "out");
const PORT = 4179;
const BASE = `http://localhost:${PORT}`;
const REF = resolve(ASSETS, "reference.mp4");
const TEST = resolve(ASSETS, "test.mp4");

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
    "npx",
    ["vite", "preview", "--port", String(PORT), "--strictPort"],
    { cwd: ROOT, stdio: "ignore" },
  );
  let browser;
  try {
    await waitForServer(BASE);
    browser = await chromium.launch({
      channel: "chrome",
      headless: true,
      args: [
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--ignore-gpu-blocklist",
        "--enable-webgl",
        "--autoplay-policy=no-user-gesture-required",
      ],
    });
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    page.on("console", (m) => {
      if (m.type() === "error") log(`page error: ${m.text()}`);
    });
    await page.goto(BASE, { waitUntil: "networkidle" });
    log("waiting for model…");
    await page.waitForFunction(
      () => /ready/i.test(document.getElementById("status")?.textContent || ""),
      null,
      { timeout: 90000 },
    );
    await page.setInputFiles("#ref-file", REF);
    await page.setInputFiles("#test-file", TEST);
    await page.waitForFunction(
      () => !document.getElementById("play-btn")?.hasAttribute("disabled"),
      null,
      { timeout: 30000 },
    );

    log("playing…");
    await page.click("#play-btn");
    // Let enough frames accumulate to draw a real curve.
    await page.waitForFunction(
      () => {
        const t = document.getElementById("score-now")?.textContent || "—";
        return t !== "—" && Number(t) > 0;
      },
      null,
      { timeout: 30000 },
    );
    await page.waitForTimeout(4000);

    // 1. Live score is a real number.
    const scoreNow = await page.$eval("#score-now", (e) => e.textContent);
    check(Number(scoreNow) > 0, "live similarity score present", `now=${scoreNow}`);

    // 2. Per-limb breakdown carries real values.
    const limbVals = await page.$$eval(".bd-val", (els) =>
      els.map((e) => e.textContent),
    );
    const realLimbs = limbVals.filter((v) => v && v !== "—").length;
    check(realLimbs >= 3, "per-limb bars populated", `${realLimbs} real values`);

    // 3. Score-history canvas has a drawn curve.
    const drawn = await page.evaluate(() => {
      const c = document.getElementById("score-history");
      const ctx = c.getContext("2d");
      const { data } = ctx.getImageData(0, 0, c.width, c.height);
      // Background is ~#fbfaf8; count pixels that differ noticeably from it.
      let nonBg = 0;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
        if (a > 10 && (Math.abs(r - 251) > 24 || Math.abs(g - 250) > 24 || Math.abs(b - 248) > 24)) {
          nonBg++;
        }
      }
      return { nonBg, w: c.width, h: c.height };
    });
    check(
      drawn.nonBg > 200,
      "score-history curve drawn",
      `${drawn.nonBg} non-bg px @ ${drawn.w}x${drawn.h}`,
    );

    // 4. Reset clears the graph.
    await page.click("#restart-btn");
    await page.waitForTimeout(300);
    const afterReset = await page.evaluate(() => {
      const c = document.getElementById("score-history");
      const ctx = c.getContext("2d");
      const { data } = ctx.getImageData(0, 0, c.width, c.height);
      let colored = 0;
      for (let i = 0; i < data.length; i += 4) {
        // Count strongly-colored (line) pixels, ignoring faint gridlines.
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        if (max - min > 60) colored++;
      }
      return colored;
    });
    check(afterReset < 50, "reset clears the curve", `${afterReset} colored px left`);
  } finally {
    if (browser) await browser.close().catch(() => {});
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
