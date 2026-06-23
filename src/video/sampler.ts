/**
 * Offline pose sampling: step a <video> through fixed time intervals, run the
 * detector at each, and return the resulting normalized-pose sequence. Used to
 * precompute the two sequences that DTW aligns.
 */

import type { PoseDetector } from "../pose/detector";
import { normalizePose, type NormalizedPose } from "../pose/normalize";

export interface SampleResult {
  /** Spacing between samples, in seconds. */
  step: number;
  /** One entry per sample (null where no person was detected). */
  poses: Array<NormalizedPose | null>;
}

/** Seek a video to `t` seconds and resolve once the frame is ready. */
function seekTo(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked);
      resolve();
    };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = t;
  });
}

/**
 * Sample `fps` poses per second from a loaded video, capped at `maxSamples`
 * frames so very long clips stay bounded. Leaves the video paused at t=0.
 */
export async function sampleVideoPoses(
  video: HTMLVideoElement,
  detector: PoseDetector,
  fps = 6,
  maxSamples = 240,
  onProgress?: (done: number, total: number) => void,
): Promise<SampleResult> {
  const duration = video.duration;
  if (!isFinite(duration) || duration <= 0) return { step: 0, poses: [] };

  let step = 1 / fps;
  let total = Math.floor(duration / step) + 1;
  if (total > maxSamples) {
    total = maxSamples;
    step = duration / Math.max(1, total - 1);
  }

  video.pause();
  const poses: Array<NormalizedPose | null> = [];
  for (let k = 0; k < total; k++) {
    const t = Math.min(duration - 1e-3, k * step);
    await seekTo(video, t);
    const pose = await detector.estimate(video);
    poses.push(pose ? normalizePose(pose) : null);
    onProgress?.(k + 1, total);
  }

  // Reset to the start so normal playback begins cleanly.
  await seekTo(video, 0);
  return { step, poses };
}

/**
 * Build a warp function mapping a reference playback time (seconds) to the
 * matched test playback time (seconds), from a DTW ref→test index map.
 */
export function buildWarp(
  refStep: number,
  testStep: number,
  refToTest: number[],
): (refTime: number) => number {
  const n = refToTest.length;
  return (refTime: number): number => {
    if (refStep <= 0 || n === 0) return refTime;
    let i = Math.round(refTime / refStep);
    i = Math.max(0, Math.min(n - 1, i));
    return refToTest[i] * testStep;
  };
}
