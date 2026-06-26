# dance-pose-coach — Task Tracker

## v0.1 — Video ⟷ Video comparison (current loop)

### Phase 1 — Skeleton (manual)
- [x] `git init`
- [x] Project scaffold: `package.json`, `tsconfig.json`, `vite.config.ts`
- [x] `.gitignore`, `LICENSE` (MIT), `README.md`
- [x] `tasks/todo.md`
- [x] `gh repo create dance-pose-coach --private` + first push

### Phase 2 — Implementation (harness loop)
- [x] `src/pose/keypoints.ts` — COCO-17 names, skeleton edges, shared types
- [x] `src/pose/detector.ts` — MoveNet wrapper, Lightning/Thunder switch
- [x] `src/pose/normalize.ts` — hip-centered + torso-scaled normalization
- [x] `src/pose/similarity.ts` — cosine similarity + 0–100 score mapping
- [x] `src/render/skeleton.ts` — Canvas keypoint + bone overlay
- [x] `src/video/dualPlayer.ts` — synchronized dual-video playback + frame pump
- [x] `src/ui/app.ts` — wire DOM + detector + players + scoring
- [x] `src/main.ts` + `index.html` + styles — runnable demo page
- [x] `npm run build` / `typecheck` passes

## v0.1.1 — Demo + UI polish (this loop)
- [x] Fix concurrency bug: run the two MoveNet inferences sequentially
      (one detector instance can't run two estimates at once)
- [x] Redesign UI to a light, premium-minimal theme (qrod.ai-inspired):
      off-white canvas, charcoal type, Inter/geometric sans, restrained borders
- [x] Add favicon (removes the only page 404)
- [x] Fully-automated demo pipeline (`npm run demo`):
      - [x] `demo/prepare-assets.mjs` — download CC BY 3.0 clip, derive ref/test
      - [x] `demo/record.mjs` — Playwright + headless Chrome, record + trim
      - [x] ffmpeg → `dance-pose-coach-demo.mp4` + README `.gif`
- [x] Embed demo GIF + CC-BY attribution in README
- [x] Verified recording shows live skeletons + score (~99) on both clips

## v0.2 — Backlog features (this loop, issues #1–#3)
- [x] **Webcam live-follow mode** (#1) — `src/video/webcam.ts`; DualPlayer
      `setLiveTest()` skips test-seeking for the live stream; UI test-source
      toggle with graceful permission fallback
- [x] **DTW timeline alignment** (#2) — `src/pose/dtw.ts` (banded DTW) +
      `src/video/sampler.ts` (offline pose sampling + warp builder); DualPlayer
      `setWarp()`; "DTW align" toggle, invalidated on clip/source change
- [x] **Per-joint breakdown** (#3) — `src/pose/perJoint.ts` (per-limb divergence
      + EMA tracker); breakdown panel + worst-limb red highlight on the skeleton
- [x] Verified: typecheck + build green; headless run shows score 99 / avg 98.8,
      DTW enabled, all five limb bars populated (right leg flagged worst)

## v0.3 — Backlog features (this loop, issues #4–#6)
- [x] **Score history graph** (#4) — `src/render/scoreGraph.ts` (DPI-aware
      sparkline, raw + EMA traces, 50/75/100 grid, hue-coded); scoreboard canvas
      + `resetScore()` helper clears it on reset / clip / source / DTW change.
      Reusable headless check added: `demo/verify.mjs` (`npm run verify`).
- [x] **Streaming DTW for webcam** (#5) — `src/pose/streamDtw.ts`
      (`StreamingAligner`: windowed best-match over recent reference poses with a
      monotonic non-decreasing matched time = lag compensation). Alignment button
      becomes "Live sync" in webcam mode; scores against the matched pose and
      shows the detected lag (ms behind). Verified with Chrome's fake camera
      (test clip piped in) — `demo/verify.mjs` scenario B.
- [x] **Export scored comparison clip** (#6) — `src/render/composite.ts`
      (side-by-side panes + skeletons + score banner) fed by a transform-aware
      `strokePose` refactored out of `skeleton.ts`; `src/video/recorder.ts`
      (`ComparisonRecorder` = MediaRecorder over `canvas.captureStream`) → webm
      download. Export button records during playback, auto-saves on Stop / clip
      end, works for file + webcam. Verified (scenario C): downloads a valid
      1280×596 webm with real frames.

## dance-pose-coach v0.5 — post-run improvement report (#9)
- [x] **`src/pose/report.ts`** — `RunRecorder` accumulates per-bone reference-vs-test
      unit-direction vectors against the aligned reference timeline; `buildReport`
      buckets into fixed time windows (default 3s), ranks limbs per segment by mean
      bone-direction error (degrees), and surfaces the run-wide top-N limb×segment
      opportunities. `correctionFor` turns the *signed* averaged bone delta into a
      directional fix ("Raise your left elbow…", "Level your shoulders").
- [x] **`src/render/reportPanel.ts`** — renders "Biggest opportunities" + a
      segment-by-segment table; each row seeks both videos to that moment.
- [x] **`DualPlayer.seek(refTime)`** — clean scrub (respects DTW warp) for row clicks.
- [x] **Wiring** (`src/ui/app.ts`, `index.html`, `style.css`) — record each scored
      frame; emit the report on run end; clear on any reset.
- [x] **Verified** — deterministic `verifyReport()` in `demo/verify.mjs` (segment +
      rank + directional coaching + empty-input safety) **and** browser scenario A
      (report shown after the run, per-segment rows, directional correction text,
      reset hides it). Validated against `reference.mp4`/`test.mp4`; same module is
      quality-tier agnostic (`swan-lake-{hq,mq,lq}` use the identical path).
- Decisions (self-grilled, AUTO mode): fixed time windows over audio beat/onset for
  v1 (no audio decode today); error in bone-direction **degrees** (amplitude-
  normalized, so fast segments aren't over-penalized); coaching from signed delta
  on the worst bone → screen-relative raise/lower/left/right + torso lean/level.

## Backlog — next loop
- [ ] Music-aware segmentation: real beat/onset detection vs fixed time windows (#9 follow-up)
- [ ] Report: extension cues ("fully extend the knee") from joint-angle magnitude, not just direction (#9 follow-up)
- [ ] Trim/scrub the exported clip range before saving (currently whole routine)
- [ ] Score history: mark the worst moments / scrub to them
- [ ] Per-rep segmentation (detect repeats in the routine)

## Review notes
- Default alignment is by playback progress; enable **DTW align** (#2) when the
  two clips differ in tempo so frames are matched by pose instead.
- MoveNet weights load from the TF Hub CDN at runtime (needs network on first
  run; cached by the browser afterward).

## dance-pose-coach v0.4 — deferred / optional (post-v0.4)
- [ ] **Real clap-audio onset detection** for sync calibration. v0.4 ships a working transport-delay estimator (`syncCalib.ts`) + countdown/clap UI, but the UI currently feeds it a synthetic pipeline-latency fallback (MediaTrackSettings.latency or ~120ms). Replace with real Web Audio onset detection of the clap to measure true end-to-end transport delay. (from eng-C1 review)
- [ ] **True wide-angle lens de-distortion** (camera-intrinsics / checkerboard calibration). v0.4 handles viewpoint via BlazePose 3D + Procrustes; real lens distortion correction is a further optional layer. (from arch-1 / Issue B)
