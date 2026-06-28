import { defineConfig } from "vite";

// MoveNet weights are fetched from the TF Hub CDN at runtime, so no special
// asset handling is required here. We keep the config minimal and rely on
// Vite's defaults for a TS + ESM project.
export default defineConfig({
  base: "./",
  // Bake an optionally-exported OpenAI key into the bundle so the dancer doesn't
  // have to paste it into the panel each session. `export DANCE_COACH_OPENAI_KEY=…`
  // before `npm run dev`/`build` and the OpenAI provider picks it up automatically;
  // unset, it compiles to "" and the provider stays opt-in (falls back as before).
  // Still browser-only — there's no server; the literal lives in the client JS,
  // which is gitignored under dist/.
  define: {
    __DPC_OPENAI_KEY__: JSON.stringify(process.env.DANCE_COACH_OPENAI_KEY ?? ""),
  },
  server: {
    port: 5173,
    open: true,
  },
  build: {
    target: "es2021",
    sourcemap: true,
  },
});
