# Lessons

## "Record a new demo" / "make a demo" → run the repo's own pipeline, don't claim you can't

**Mistake (2026-06-25):** When Eddie said "Record a new demo", I replied that I "don't have a tool that records demos (no screen/video capture here)" and asked what he meant — twice. This was wrong: I had already recorded this exact demo multiple times *in the same chat* using this repo's `npm run demo` pipeline (Playwright headless Chrome → ffmpeg MP4/GIF).

**Why it happened:** Context for this gateway session is summarized/truncated, so the earlier "demo done" turns weren't in my visible window. I treated "record a demo" as a generic capability question instead of recognizing it as a recurring, established task for the active project.

**How to apply:**
- For a terse recurring request ("record a new demo", "make a demo"), first assume it refers to the **active project's established workflow**. Search chat history (`search_chat_history`) before declaring a capability missing.
- The demo workflow for `dance-pose-coach`: `npm run demo` (build + `demo/record.mjs`, headless Chrome, no display needed) → outputs `demo/dance-pose-coach-demo.{mp4,gif}`. Verify with frame/score checks, not by assertion.
- Delivery: the bot's Telegram/Gmail tools can't attach binaries, so publish artifacts as a **GitHub Release** and put direct-download links in the README and (if asked) a Gmail draft.
- General rule: never report a capability as unavailable without first checking whether I've already done it in this chat/project.
