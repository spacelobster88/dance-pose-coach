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

## Backlog — next loop
- [ ] Webcam live-follow mode (detection pipeline already shared)
- [ ] DTW timeline alignment (handle differing tempo between clips)
- [ ] Per-joint breakdown (which limb diverged most)
- [ ] Score smoothing / EMA + on-screen score history graph
- [ ] Export/record a scored comparison clip

## Review notes
- Alignment is by playback progress only in v0.1 — clips should be similar
  length and start on the same beat for meaningful scores.
- MoveNet weights load from the TF Hub CDN at runtime (needs network on first
  run; cached by the browser afterward).
