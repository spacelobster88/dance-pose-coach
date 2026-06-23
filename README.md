# dance-pose-coach

Video-vs-video dance pose comparison coach. Upload a **reference video** (the
"correct" choreography) and a **test video** (your attempt), play them side by
side, and get a real-time pose-similarity score driven by
[MoveNet](https://www.tensorflow.org/hub/tutorials/movenet) running entirely in
the browser via TensorFlow.js.

> **Status:** the v0.1 video ⟷ video core now also ships **webcam live-follow**,
> **DTW timeline alignment**, and a **per-limb divergence breakdown** (issues
> [#1](../../issues/1)–[#3](../../issues/3)). See [`tasks/todo.md`](tasks/todo.md).

![dance-pose-coach demo](demo/dance-pose-coach-demo.gif)

> Demo footage derived from "Squat - exercise demonstration video" by
> FitnessScape, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0). The
> right clip is the same movement zoomed + time-shifted, so the score stays high
> (normalization handles zoom/position) while still reacting to the timing
> drift. Regenerate anytime with `npm run demo` — see [`demo/`](demo/).

## Features (v0.1)

1. **MoveNet detection** — 17 keypoints per frame; switch between
   `Lightning` (fast) and `Thunder` (accurate) at runtime.
2. **Skeleton overlay** — keypoints + bone edges drawn on a Canvas over each
   video (target ≥30fps on the Lightning model).
3. **Pose normalization** — hip-centered, torso-scaled so the score is
   invariant to where the dancer stands and how large they appear in frame.
4. **Cosine-similarity scoring** — per-frame similarity between reference and
   test poses, surfaced as a live 0–100 score plus a running average.
5. **Dual side-by-side playback** — both videos share a single play/pause and
   seek so frames stay aligned by playback progress.
6. **Runnable demo page** — no build step needed to try it; just `npm run dev`.

## Added since v0.1

7. **Webcam live-follow** (#1) — switch the test source from a file to the live
   camera (`getUserMedia`) and follow along against a reference routine in real
   time. The detection/scoring pipeline is identical; only the frame source
   changes. Falls back gracefully if permission is denied.
8. **DTW timeline alignment** (#2) — the **DTW align** button samples both clips
   into pose sequences and runs banded Dynamic Time Warping so a slower/faster
   attempt is matched to the reference *by pose* instead of by raw progress.
   Toggle off to return to linear alignment.
9. **Per-limb divergence breakdown** (#3) — a panel under the score ranks how far
   each limb (arms, legs, torso) is from the reference, and the worst-diverging
   limb is highlighted in red on the skeleton overlay so you know what to fix.

## Quick start

```bash
npm install
npm run dev      # opens http://localhost:5173
```

Then in the browser:

1. Pick a **Reference video** and a **Test video** (any `mp4`/`webm`/`mov`).
2. Choose the model (Lightning is the default; Thunder is more accurate but
   slower).
3. Press **Play** — both videos play together, skeletons overlay in real time,
   and the similarity score updates each frame.

### Build for production

```bash
npm run build    # type-checks then emits a static bundle to dist/
npm run preview  # serve the built bundle locally
```

## How scoring works

For each video frame we:

1. Run MoveNet to get 17 `(x, y, score)` keypoints.
2. Drop low-confidence keypoints, then **normalize**: translate so the mid-hip
   sits at the origin and scale by torso length (shoulder-to-hip distance).
3. Flatten the kept, shared keypoints into a vector and compute **cosine
   similarity** between the reference vector and the test vector.
4. Map similarity `[-1, 1] → [0, 100]` for display.

By default alignment is by **playback progress** (not content), so keep the two
clips roughly the same length and starting on the same beat. When tempos differ,
turn on **DTW align**: both clips are sampled into normalized-pose sequences and
matched frame-to-frame by pose distance `(1 − cosine)/2`, so the score reflects
*how* you moved rather than *when*. The **per-limb breakdown** then decomposes
the residual difference by body part.

## Tech stack

- **Vite** + **TypeScript** (ESM, no framework)
- **@tensorflow-models/pose-detection** (MoveNet)
- **@tensorflow/tfjs-backend-webgl** for GPU-accelerated inference

## Project layout

```
src/
  pose/
    detector.ts     # MoveNet wrapper, Lightning/Thunder switch
    normalize.ts    # hip-center + torso-scale normalization
    similarity.ts   # cosine similarity + score mapping
    dtw.ts          # banded DTW alignment over pose sequences (#2)
    perJoint.ts     # per-limb divergence + worst-limb tracking (#3)
    keypoints.ts    # COCO-17 names, skeleton edges, types
  render/
    skeleton.ts     # Canvas skeleton drawing (+ limb highlight)
  video/
    dualPlayer.ts    # synchronized two-video playback + frame pump (+ warp/live)
    sampler.ts       # offline pose sampling + warp builder for DTW (#2)
    webcam.ts        # getUserMedia capture for live-follow (#1)
  ui/
    app.ts           # wires DOM, detector, players, scoring together
  main.ts            # entry point
index.html
```

## License

MIT — see [LICENSE](LICENSE).
