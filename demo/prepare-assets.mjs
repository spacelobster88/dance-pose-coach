// Reproducibly prepare the two demo clips used by record.mjs.
//
//   node demo/prepare-assets.mjs   (or: npm run demo:assets)
//
// Downloads a CC BY 3.0 single-subject exercise clip from Wikimedia Commons and
// derives:
//   - reference.mp4 : the "correct" choreography (trimmed to 7s)
//   - test.mp4      : the "attempt" — same movement, zoomed + shifted + offset
//                     0.3s in time, so the live score stays high (normalization
//                     handles zoom/position) yet reacts to the timing drift.
//
// Source: "Squat - exercise demonstration video" by FitnessScape, CC BY 3.0
// https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ASSETS = resolve(__dirname, "assets");
const SOURCE = resolve(ASSETS, "source_squat.webm");
const REF = resolve(ASSETS, "reference.mp4");
const TEST = resolve(ASSETS, "test.mp4");

const FILE_TITLE = "File:Squat_-_exercise_demonstration_video.webm";

function log(m) {
  console.log(`[assets] ${m}`);
}

function ff(args) {
  const r = spawnSync("ffmpeg", ["-y", ...args], { stdio: "inherit" });
  if (r.status !== 0) throw new Error(`ffmpeg failed: ${args.join(" ")}`);
}

async function resolveUploadUrl() {
  const api =
    "https://commons.wikimedia.org/w/api.php?action=query&format=json" +
    `&titles=${encodeURIComponent(FILE_TITLE)}&prop=imageinfo&iiprop=url`;
  const res = await fetch(api, {
    headers: { "User-Agent": "dance-pose-coach-demo/0.1 (educational)" },
  });
  const data = await res.json();
  const page = Object.values(data.query.pages)[0];
  const url = page?.imageinfo?.[0]?.url;
  if (!url) throw new Error("could not resolve Wikimedia upload URL");
  return url;
}

async function main() {
  mkdirSync(ASSETS, { recursive: true });

  if (!existsSync(SOURCE)) {
    const url = await resolveUploadUrl();
    log(`downloading source clip…\n  ${url}`);
    const res = await fetch(url, {
      headers: { "User-Agent": "dance-pose-coach-demo/0.1 (educational)" },
    });
    if (!res.ok) throw new Error(`download failed: HTTP ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    writeFileSync(SOURCE, buf);
    log(`saved source (${(buf.length / 1024 / 1024).toFixed(1)} MB)`);
  } else {
    log("source already present, skipping download");
  }

  log("building reference.mp4 (7s, H.264)…");
  ff([
    "-i", SOURCE,
    "-t", "7.0",
    "-an",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "20",
    "-vf", "scale=1280:720",
    REF,
  ]);

  log("building test.mp4 (zoom 1.12x + shift + 0.3s offset)…");
  ff([
    "-ss", "0.3",
    "-i", SOURCE,
    "-t", "6.8",
    "-an",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "20",
    "-vf", "scale=1434:806,crop=1280:720:120:40",
    TEST,
  ]);

  log(`done:\n  ${REF}\n  ${TEST}`);
}

main().catch((e) => {
  console.error("[assets] FAILED:", e);
  process.exit(1);
});
