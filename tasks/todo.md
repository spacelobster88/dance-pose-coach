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

## Performance: memory bloat + page jank + score latency (#14)
Root cause: the whole pipeline (2× MoveNet inference + normalize + similarity +
per-limb + draw) ran on the main thread **every rAF** (~60–120fps), on the
**full-resolution** `<video>`. Main-thread saturation → jank + score lag, and the
constant full-res GPU texture uploads → memory growth.

- [x] **Throttle detection, decoupled from render** (`src/video/dualPlayer.ts`)
      — the rAF render loop is replaced with a tick scheduler that prefers
      `requestVideoFrameCallback` (infer **once per decoded frame**, naturally
      throttled to the clip's ~24–30fps), falling back to rAF capped at ~24fps.
      Drift correction still runs every tick; the existing `frameBusy` guard
      drops any tick that lands mid-inference so detection never overlaps.
- [x] **Downscale the detector input** (`src/pose/detector.ts`) — the source is
      drawn into a single **reused** offscreen canvas (256px longest side, the
      MoveNet path only) before `estimatePoses`; keypoints are mapped back into
      source-pixel space so every downstream consumer (skeleton, normalize,
      composite) is byte-identical. A 1280×720 clip uploads a 256×144 tensor
      instead of full-res — a ~25× smaller per-frame GPU upload.
- [x] **Memory hygiene** — `WEBGL_DELETE_TEXTURE_THRESHOLD=0` so detection
      textures are freed eagerly instead of pooled (keeps GPU memory flat across
      a clip); single reused scratch canvas; detector loaded once (already true).
- [x] **Score-UI cadence decoupled** — detection now runs at ≤ decoded-frame
      rate (not 60–120fps), so the score/breakdown DOM updates are throttled to
      detection automatically without extra UI-side gating.
- [ ] **Deferred: Web Worker + OffscreenCanvas inference offload.** The issue
      lists this as one proposed mechanism; it's a large, higher-risk refactor
      (touches the DTW sampler, live-sync ordering, recording, and the
      swiftshader `verify.mjs` harness). The behavioral acceptance — no sustained
      jank, score within ~1 frame, flat memory — is met by the throttle +
      downscale + texture hygiene above, so the worker is left as a follow-up.
      Verified: `npm run verify` ALL CHECKS PASSED (Chrome took the rVFC path).

## v0.5 — AI coaching insights (#15)
- [x] **Model-agnostic insights layer** — `src/insights/`:
      - [x] `types.ts` — `CoachingInput` (segments + per-limb degrees + signed
            direction + score series + top-N opportunities) + `CoachingProvider`
            (`generateCoaching(report)`) pluggable interface.
      - [x] `report.ts` — `RunReport` aggregator that compiles the structured
            report from the existing per-frame score + per-limb divergence:
            angular error in **degrees** (bone vectors) and a **signed**
            correction direction (mean keypoint offset), segmented over time with
            top-3 opportunities. (This is #9's "rule-based report", which did not
            actually exist in-tree yet — built here.)
      - [x] `prompt.ts` — dance-coach system prompt + compact report serializer.
      - [x] `ruleBased.ts` — deterministic offline coaching (the fallback) +
            provider wrapper.
      - [x] `ollama.ts` — local Ollama provider (`/api/chat`, NDJSON streaming),
            recommended default; availability-probes `/api/tags`.
      - [x] `claude.ts` — **opt-in** remote provider via a user-configured thin
            proxy (key stays server-side); default model `claude-opus-4-8`;
            SSE/JSON/text responses.
      - [x] `markdown.ts` — tiny zero-dependency Markdown → HTML (escapes first).
      - [x] `storage.ts` — safe localStorage accessor (no-op off-browser).
      - [x] `index.ts` — provider registry + `generateCoaching()` dispatcher with
            **Auto** (local-if-available) and **clean fallback** to rule-based on
            unavailability/error.
- [x] **UI** — `index.html` "✨ AI coaching" panel (provider select + contextual
      Ollama-model / Claude-proxy-URL inputs, persisted in localStorage) +
      coaching panel; `src/ui/app.ts` feeds the `RunReport` each scored frame,
      streams tokens into the panel, renders Markdown, and surfaces the provider
      / fallback note. Privacy-preserving: only derived stats leave the page, and
      only on the opt-in remote path.
- [x] **Tests** — `npm test` (`test/insights.test.ts`, runs real TS via a tiny
      `--import` resolver hook): report aggregation, rule-based output, Markdown,
      and dispatcher fallback (23 assertions). `demo/verify.mjs` scenario A also
      asserts the offline coaching panel renders Markdown end-to-end in Chrome.
- [x] `npm run typecheck` + `npm run build` green.

## Backlog — next loop
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
