import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

import { KEYPOINT_INDEX, type Pose } from "./keypoints";
import { buildHistogram, poseBBox } from "./tracker";

/**
 * Model selector. The two MoveNet variants are the DEFAULT 2D path; "blazepose"
 * is an opt-in 3D path whose weights load from a CDN at runtime (not bundled).
 */
export type ModelVariant = "lightning" | "thunder" | "blazepose";

/** CDN hosting the MediaPipe Pose wasm + model assets, loaded at runtime. */
const BLAZEPOSE_SOLUTION_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/pose";

type MoveNetModelType =
  (typeof poseDetection.movenet.modelType)[keyof typeof poseDetection.movenet.modelType];

const MOVENET_MODEL_TYPE: Record<"lightning" | "thunder", MoveNetModelType> = {
  lightning: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
  thunder: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
};

function isBlazePose(v: ModelVariant): boolean {
  return v === "blazepose";
}

/** Width/height of the source media for canvas sampling, or null if unknown. */
function sourceSize(
  source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
): { w: number; h: number } | null {
  const w =
    (source as HTMLVideoElement).videoWidth ||
    (source as HTMLCanvasElement).width ||
    (source as HTMLImageElement).naturalWidth ||
    0;
  const h =
    (source as HTMLVideoElement).videoHeight ||
    (source as HTMLCanvasElement).height ||
    (source as HTMLImageElement).naturalHeight ||
    0;
  return w > 0 && h > 0 ? { w, h } : null;
}

/**
 * Sample a torso color histogram for each detection by reading the torso region
 * (shoulders→hips bbox) from the source via a scratch 2D canvas. Gracefully
 * no-ops — returning all-undefined — when no 2D canvas/pixels are available
 * (Node/test), in which case appearance stays undefined and the tracker falls
 * back to motion-only matching. Never throws.
 */
function sampleTorsoHistograms(
  source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
  poses: Pose[],
): Array<number[] | undefined> {
  const none: Array<number[] | undefined> = poses.map(() => undefined);
  // No DOM / no canvas factory (Node, SSR) → bail.
  if (typeof document === "undefined" || typeof document.createElement !== "function") {
    return none;
  }
  const size = sourceSize(source);
  if (!size) return none;
  let ctx: CanvasRenderingContext2D | null;
  let canvas: HTMLCanvasElement;
  try {
    canvas = document.createElement("canvas");
    canvas.width = size.w;
    canvas.height = size.h;
    ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return none;
    ctx.drawImage(source as CanvasImageSource, 0, 0, size.w, size.h);
  } catch {
    return none; // tainted canvas / unsupported source → motion-only
  }

  const LS = KEYPOINT_INDEX.left_shoulder;
  const RS = KEYPOINT_INDEX.right_shoulder;
  const LH = KEYPOINT_INDEX.left_hip;
  const RH = KEYPOINT_INDEX.right_hip;

  return poses.map((pose) => {
    // Prefer a tight torso box (shoulders→hips); fall back to the pose bbox.
    const k = pose.keypoints;
    const torso = [k[LS], k[RS], k[LH], k[RH]].filter((p) => p && (p.score ?? 0) >= 0.3);
    let box: { minX: number; minY: number; maxX: number; maxY: number } | null = null;
    if (torso.length >= 3) {
      const xs = torso.map((p) => p.x);
      const ys = torso.map((p) => p.y);
      box = { minX: Math.min(...xs), minY: Math.min(...ys), maxX: Math.max(...xs), maxY: Math.max(...ys) };
    } else {
      box = poseBBox(pose);
    }
    if (!box) return undefined;
    const x0 = Math.max(0, Math.floor(box.minX));
    const y0 = Math.max(0, Math.floor(box.minY));
    const x1 = Math.min(size.w, Math.ceil(box.maxX));
    const y1 = Math.min(size.h, Math.ceil(box.maxY));
    const bw = x1 - x0;
    const bh = y1 - y0;
    if (bw <= 1 || bh <= 1) return undefined;
    try {
      const img = ctx!.getImageData(x0, y0, bw, bh).data; // RGBA
      // Flatten to RGB, skipping fully-transparent pixels.
      const rgb: number[] = [];
      for (let i = 0; i + 3 < img.length; i += 4) {
        if (img[i + 3] < 8) continue;
        rgb.push(img[i], img[i + 1], img[i + 2]);
      }
      if (rgb.length < 3) return undefined;
      const hist = buildHistogram(rgb);
      return hist.some((v) => v > 0) ? hist : undefined;
    } catch {
      return undefined;
    }
  });
}

/** Map a raw detector pose into our COCO-17 `Pose` (2D, plus 3D when present). */
function toPose(p: poseDetection.Pose): Pose {
  const pose: Pose = {
    keypoints: p.keypoints.map((k) => ({
      x: k.x,
      y: k.y,
      z: k.z ?? undefined,
      score: k.score ?? undefined,
      name: k.name,
    })),
    score: p.score ?? undefined,
  };
  // 3D world landmarks are only present on the BlazePose path.
  if (p.keypoints3D && p.keypoints3D.length) {
    pose.worldKeypoints = p.keypoints3D.map((k) => ({
      x: k.x,
      y: k.y,
      z: k.z ?? undefined,
      score: k.score ?? undefined,
      name: k.name,
    }));
  }
  return pose;
}

/**
 * Thin wrapper around single-pose detection.
 *
 * Holds one detector instance and lets the UI hot-swap between the MoveNet
 * Lightning (fast) / Thunder (accurate) 2D variants and the opt-in BlazePose
 * 3D variant without tearing down the whole app. When BlazePose is active the
 * estimate carries 3D world landmarks; the 2D mapping is unchanged for MoveNet.
 */
export class PoseDetector {
  private detector: poseDetection.PoseDetector | null = null;
  private variant: ModelVariant;
  private loadingPromise: Promise<void> | null = null;
  private backendReady = false;
  // Multi-person path: a separate, lazily-loaded MoveNet MultiPose detector.
  // Kept independent of the single-pose `detector` so the variant selector
  // (lightning / thunder / 3D BlazePose) and its behavior are untouched.
  private multiDetector: poseDetection.PoseDetector | null = null;
  private multiLoading: Promise<void> | null = null;

  constructor(variant: ModelVariant = "lightning") {
    this.variant = variant;
  }

  get currentVariant(): ModelVariant {
    return this.variant;
  }

  /** True when the active variant emits 3D world landmarks. */
  get is3D(): boolean {
    return isBlazePose(this.variant);
  }

  /**
   * Ensure the WebGL backend is selected and ready exactly once.
   * Only relevant to the MoveNet (tfjs) path — the BlazePose mediapipe runtime
   * runs in its own wasm solution and needs no tf backend.
   */
  private async ensureBackend(): Promise<void> {
    if (this.backendReady) return;
    await tf.setBackend("webgl");
    await tf.ready();
    this.backendReady = true;
  }

  /** Load (or reload) the model for the current variant. */
  async load(): Promise<void> {
    if (this.loadingPromise) return this.loadingPromise;
    this.loadingPromise = (async () => {
      if (isBlazePose(this.variant)) {
        // MediaPipe runtime: no tf backend; weights stream from the CDN.
        this.detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.BlazePose,
          {
            runtime: "mediapipe",
            modelType: "full",
            enableSmoothing: true,
            solutionPath: BLAZEPOSE_SOLUTION_PATH,
          },
        );
        return;
      }
      await this.ensureBackend();
      // Narrowed: only "lightning" | "thunder" reach this branch.
      const movenetVariant = this.variant as "lightning" | "thunder";
      this.detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: MOVENET_MODEL_TYPE[movenetVariant],
          enableSmoothing: true,
        },
      );
    })();
    try {
      await this.loadingPromise;
    } finally {
      this.loadingPromise = null;
    }
  }

  /** Switch model variant, disposing the previous detector first. */
  async setVariant(variant: ModelVariant): Promise<void> {
    if (variant === this.variant && this.detector) return;
    this.variant = variant;
    const old = this.detector;
    this.detector = null;
    old?.dispose();
    await this.load();
  }

  /**
   * Estimate a single pose from a video/image/canvas source.
   * Returns null when no person is detected. For BlazePose, `worldKeypoints`
   * (and per-keypoint `z`) carry the 3D landmarks; for MoveNet the 2D mapping
   * is byte-for-byte the original behavior.
   */
  async estimate(
    source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
  ): Promise<Pose | null> {
    if (!this.detector) {
      throw new Error("PoseDetector.estimate called before load()");
    }
    const poses = await this.detector.estimatePoses(source, {
      maxPoses: 1,
      flipHorizontal: false,
    });
    if (!poses.length) return null;
    return toPose(poses[0]);
  }

  /**
   * Lazily create the MoveNet MultiPose detector used by the multi-person mode.
   * Idempotent and concurrency-safe (one in-flight load is shared).
   */
  async loadMulti(): Promise<void> {
    if (this.multiDetector) return;
    if (this.multiLoading) return this.multiLoading;
    this.multiLoading = (async () => {
      await this.ensureBackend();
      this.multiDetector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
          enableSmoothing: true,
        },
      );
    })();
    try {
      await this.multiLoading;
    } finally {
      this.multiLoading = null;
    }
  }

  /**
   * Estimate up to `maxPoses` people from a source (MoveNet MultiPose). The
   * returned poses are unordered and identity-free; the tracker assigns stable
   * ids. Loads the multipose model on first use.
   */
  async estimateMany(
    source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
    maxPoses = 6,
  ): Promise<Pose[]> {
    await this.loadMulti();
    const raw = await this.multiDetector!.estimatePoses(source, {
      maxPoses,
      flipHorizontal: false,
    });
    const poses = raw.map(toPose);
    // Attach a torso color histogram per detection (DeepSORT-lite appearance
    // cue) when a 2D canvas is available; a graceful no-op headless.
    const hists = sampleTorsoHistograms(source, poses);
    for (let i = 0; i < poses.length; i++) {
      if (hists[i]) poses[i].appearance = hists[i];
    }
    return poses;
  }

  dispose(): void {
    this.detector?.dispose();
    this.detector = null;
    this.multiDetector?.dispose();
    this.multiDetector = null;
  }
}
