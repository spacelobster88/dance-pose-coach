# Demo

A fully-automated, reproducible recording of `dance-pose-coach` running the
video ⟷ video comparison end to end.

## Regenerate it

```bash
npm run demo
```

This will (all without any manual interaction):

1. `npm run build` the app.
2. `node demo/prepare-assets.mjs` — download a CC-licensed single-subject clip
   and derive `reference.mp4` + `test.mp4` (skipped if they already exist).
3. `node demo/record.mjs` — launch headless Chrome via Playwright, load the two
   clips, wait for MoveNet, play them, and record the viewport.
4. `ffmpeg` encodes the result into:
   - `demo/dance-pose-coach-demo.mp4` (trimmed to the playback portion)
   - `demo/dance-pose-coach-demo.gif` (README-friendly, ~3 MB)

Just the clips, no recording:

```bash
npm run demo:assets
```

## How the test clip is built

The `test.mp4` is the **same movement** as `reference.mp4` but **zoomed ~1.12×,
shifted, and offset 0.3s in time**. This is deliberate: it shows the score
staying high (the hip-centered, torso-scaled normalization is invariant to zoom
and position) while still reacting to the timing drift — i.e. the pipeline is
genuinely comparing poses, not just echoing identical frames.

## Source footage attribution

The demo clips are derived from:

> **"Squat - exercise demonstration video"** by **FitnessScape**,
> licensed **CC BY 3.0**.
> Source: <https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm>

Per CC BY 3.0, the derived `reference.mp4`, `test.mp4`, and the recorded
`.mp4`/`.gif` retain this attribution. No endorsement by the author is implied.

## Notes

- Headless Chrome renders WebGL via ANGLE/SwiftShader, so no display server or
  GPU is required — it runs on a headless machine.
- `demo/out/` holds intermediate artifacts (raw webm, palette) and is ignored.
