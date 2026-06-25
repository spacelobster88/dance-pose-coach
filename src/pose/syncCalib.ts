// One-time sync calibration for the live webcam pipeline (Issue C, eng-C1).
//
// A live camera feed isn't instantaneous: between a real-world event and the
// moment its pose is scored, there is fixed *machine* latency — camera capture,
// USB/encode, browser decode, and pose inference. We call this the **transport
// delay**. It is a property of the *setup*, not the dancer.
//
// This module turns a one-time calibration event (a countdown that hits zero, or
// a clap emitted at a known instant) into an estimate of that transport delay,
// kept strictly separate from the human's *reaction* lag (how late the dancer is,
// which StreamingAligner measures continuously).
//
// The model is deliberately tiny and framework-free so it is trivially testable:
//   - A calibration "sample" pairs the time a cue was EMITTED (countdown zero /
//     clap played) with the time that same cue was OBSERVED back through the
//     capture pipeline (e.g. the impulse detected in the decoded webcam frame).
//   - transport delay = observed - emitted, per sample.
//   - With several noisy samples we take the median for robustness, then clamp to
//     >= 0 (a negative or implausible transport delay is unphysical).
//
// The thin UI hook (app.ts) feeds the resulting offset to StreamingAligner via the
// seam exported here; eng-C2 reads it.

/** A single calibration observation: a cue emitted at `emittedMs`, observed back
 *  through the capture pipeline at `observedMs` (both on the same clock). */
export interface CalibSample {
  /** Time the cue was emitted (countdown hit zero / clap played), ms. */
  emittedMs: number;
  /** Time that cue was observed back through the pipeline, ms. */
  observedMs: number;
}

/** Result of estimating transport delay from one or more calibration samples. */
export interface TransportEstimate {
  /** Estimated transport (machine) delay in ms, clamped to >= 0. */
  transportMs: number;
  /** Number of samples that contributed to the estimate. */
  samples: number;
  /** False when there were no samples (transportMs falls back to 0). */
  ok: boolean;
}

/** Transport delay for a single sample: observed - emitted, clamped to >= 0.
 *  A non-finite or negative raw delay (clock skew / mis-detection) clamps to 0. */
export function transportDelay(emittedMs: number, observedMs: number): number {
  const raw = observedMs - emittedMs;
  if (!Number.isFinite(raw) || raw < 0) return 0;
  return raw;
}

/** Median of a numeric list (mean of the middle two for even length).
 *  Returns 0 for an empty list. */
function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

/**
 * Estimate end-to-end transport delay from one or more calibration samples.
 *
 * Each sample contributes `observed - emitted`; we take the **median** across
 * samples so a single mis-detected clap doesn't skew the result, then clamp the
 * estimate to >= 0 (negative / implausible transport delay is unphysical).
 *
 * With no samples the estimate is a safe **0 ms** (`ok: false`), so an uncalibrated
 * setup behaves exactly as before — fully backward compatible.
 */
export function estimateTransport(samples: CalibSample[]): TransportEstimate {
  if (samples.length === 0) {
    return { transportMs: 0, samples: 0, ok: false };
  }
  const deltas = samples.map((s) => transportDelay(s.emittedMs, s.observedMs));
  const transportMs = Math.max(0, median(deltas));
  return { transportMs, samples: samples.length, ok: true };
}

/**
 * Separate a measured *total* lag into its machine (transport) and human
 * (reaction) parts: reaction = totalLag - transport, clamped to >= 0.
 *
 * StreamingAligner reports total observed lag; subtracting the calibrated
 * transport delay leaves the delay-compensated reaction lag attributable to the
 * dancer alone. Clamped so a transport estimate slightly larger than the measured
 * lag never produces a negative reaction figure.
 */
export function reactionLag(totalLagMs: number, transportMs: number): number {
  const r = totalLagMs - transportMs;
  if (!Number.isFinite(r) || r < 0) return 0;
  return r;
}

/**
 * Shared, mutable transport-delay seam.
 *
 * eng-C1 writes the calibrated value here after a successful calibration; eng-C2
 * reads it when constructing / configuring StreamingAligner (to widen `maxLagMs`
 * and compensate the reaction lag). Defaults to 0 so nothing changes until the
 * user calibrates.
 */
let calibratedTransportMs = 0;

/** Read the current calibrated transport offset (ms). Defaults to 0. */
export function getTransportOffsetMs(): number {
  return calibratedTransportMs;
}

/** Store the calibrated transport offset (ms). Clamped to >= 0. */
export function setTransportOffsetMs(ms: number): void {
  calibratedTransportMs = Number.isFinite(ms) && ms > 0 ? ms : 0;
}
