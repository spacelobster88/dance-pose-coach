import { defineConfig } from "vite";

// MoveNet weights are fetched from the TF Hub CDN at runtime, so no special
// asset handling is required here. We keep the config minimal and rely on
// Vite's defaults for a TS + ESM project.
export default defineConfig({
  base: "./",
  server: {
    port: 5173,
    open: true,
  },
  build: {
    target: "es2021",
    sourcemap: true,
  },
});
