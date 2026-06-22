// Fully-automated demo recorder.
//
//   node demo/record.mjs            (or: npm run demo)
//
// 1. Builds are assumed done; this spins up `vite preview` on the dist build.
// 2. Drives real Chrome via Playwright: loads the two demo clips, waits for the
//    MoveNet model, plays them, and records the viewport to webm.
// 3. ffmpeg converts the webm into demo/dance-pose-coach-demo.mp4 and a looping
//    demo/dance-pose-coach-demo.gif for the README.
//
// No manual interaction, no display server required (headless Chrome).

import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { existsSync, mkdirSync, readdirSync, renameSync, rmSync } from "node:fs";
import { chromium } from "playwright";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const ASSETS = resolve(__dirname, "assets");
const OUT = resolve(__dirname, "out");
const PORT = 4178;
const BASE = `http://localhost:${PORT}`;

const REF = resolve(ASSETS, "reference.mp4");
const TEST = resolve(ASSETS, "test.mp4");

function log(msg) {
  console.log(`[demo] ${msg}`);
}

function sh(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (r.status !== 0) {
    throw new Error(`${cmd} ${args.join(" ")} -> exit ${r.status}`);
  }
}

async function waitForServer(url, timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`server at ${url} did not start in ${timeoutMs}ms`);
}

async function main() {
  if (!existsSync(REF) || !existsSync(TEST)) {
    log("demo clips missing — preparing assets…");
    sh("node", [resolve(__dirname, "prepare-assets.mjs")], { cwd: ROOT });
  }

  // Ensure a fresh dist build exists.
  if (!existsSync(resolve(ROOT, "dist/index.html"))) {
    log("no dist build found — building…");
    sh("npm", ["run", "build"], { cwd: ROOT });
  }

  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(OUT, { recursive: true });

  // Start the preview server.
  log(`starting preview server on ${BASE}…`);
  const server = spawn(
    "npx",
    ["vite", "preview", "--port", String(PORT), "--strictPort"],
    { cwd: ROOT, stdio: "ignore" },
  );

  let browser;
  let playOffsetSec = 0;
  try {
    await waitForServer(BASE);
    log("preview server is up");

    log("launching Chrome (headless, WebGL via ANGLE/SwiftShader)…");
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

    const context = await browser.newContext({
      viewport: { width: 1280, height: 864 },
      deviceScaleFactor: 1,
      recordVideo: { dir: OUT, size: { width: 1280, height: 864 } },
    });
    // Playwright starts recording at context creation; remember that instant so
    // we can trim the model-loading lead-in out of the final clip.
    const recordStart = Date.now();
    const page = await context.newPage();
    page.on("console", (m) => {
      const t = m.type();
      if (t === "error" || t === "warning") log(`page ${t}: ${m.text()}`);
    });

    log("opening app…");
    await page.goto(BASE, { waitUntil: "networkidle" });

    // Wait for the MoveNet model to report ready.
    log("waiting for MoveNet model to load…");
    await page.waitForFunction(
      () => /ready/i.test(document.getElementById("status")?.textContent || ""),
      null,
      { timeout: 90000 },
    );

    log("loading demo clips…");
    await page.setInputFiles("#ref-file", REF);
    await page.setInputFiles("#test-file", TEST);

    // Wait until the Play button is enabled (both clips + model ready).
    await page.waitForFunction(
      () => !document.getElementById("play-btn")?.hasAttribute("disabled"),
      null,
      { timeout: 30000 },
    );

    // Scroll so the two video panes + scoreboard are framed (hero scrolls off).
    await page.evaluate(() => {
      const stage = document.querySelector(".stage");
      if (stage) {
        const y = stage.getBoundingClientRect().top + window.scrollY - 24;
        window.scrollTo({ top: y, behavior: "instant" });
      }
    });
    await page.waitForTimeout(600);

    log("playing — recording comparison…");
    // Lead-in to trim later: time from record start to ~0.6s before play.
    playOffsetSec = Math.max(0, (Date.now() - recordStart) / 1000 - 0.6);
    await page.click("#play-btn");

    // Let the comparison run to completion (status flips to "Finished …").
    try {
      await page.waitForFunction(
        () =>
          /finished/i.test(
            document.getElementById("status")?.textContent || "",
          ),
        null,
        { timeout: 30000 },
      );
      log("playback finished");
    } catch {
      log("finish signal not seen in time — capturing what we have");
    }
    // Hold the final frame (showing the session average) briefly.
    await page.waitForTimeout(1500);

    log("closing context to flush video…");
    await context.close();
    await browser.close();
    browser = null;
  } finally {
    if (browser) await browser.close().catch(() => {});
    server.kill("SIGTERM");
  }

  // Find the webm Playwright just wrote.
  const webm = readdirSync(OUT)
    .filter((f) => f.endsWith(".webm"))
    .map((f) => resolve(OUT, f))[0];
  if (!webm) throw new Error("Playwright produced no video file");
  const rawWebm = resolve(OUT, "raw.webm");
  renameSync(webm, rawWebm);
  log(`raw recording: ${rawWebm}`);

  const mp4 = resolve(__dirname, "dance-pose-coach-demo.mp4");
  const gif = resolve(__dirname, "dance-pose-coach-demo.gif");

  const trim = playOffsetSec.toFixed(2);
  log(`encoding mp4 (trimming ${trim}s lead-in)…`);
  sh("ffmpeg", [
    "-y",
    "-ss", trim,
    "-i", rawWebm,
    "-vf", "scale=1280:-2:flags=lanczos",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "22",
    "-movflags", "+faststart",
    mp4,
  ]);

  log("encoding gif (README-friendly, 10fps, 800px)…");
  const palette = resolve(OUT, "palette.png");
  sh("ffmpeg", [
    "-y",
    "-ss", trim,
    "-i", rawWebm,
    "-vf", "fps=10,scale=800:-1:flags=lanczos,palettegen=max_colors=160",
    palette,
  ]);
  sh("ffmpeg", [
    "-y",
    "-ss", trim,
    "-i", rawWebm,
    "-i", palette,
    "-lavfi",
    "fps=10,scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
    gif,
  ]);

  log(`DONE\n  mp4: ${mp4}\n  gif: ${gif}`);
}

main().catch((err) => {
  console.error("[demo] FAILED:", err);
  process.exit(1);
});
