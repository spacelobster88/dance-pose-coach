import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

import type { Pose } from "./keypoints";

/**
 * Model selector. The two MoveNet variants are the DEFAULT 2D path; "blazepose"
 * is an opt-in 3D path whose weights load from a CDN at runtime (not bundled).
 */
export type ModelVariant = "lightning" | "thunder" | "blazepose";

/** CDN hosting the MediaPipe Pose wasm + model assets, loaded at runtime. */
const BLAZEPOSE_SOLUTION_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/pose";

/**
 * Longest-side, in pixels, of the image actually handed to MoveNet. The model
 * resizes its input to a tiny tensor internally (192px Lightning / 256px
 * Thunder), so feeding it a full-resolution <video> only wastes a large GPU
 * texture upload and the memory backing it every frame. Drawing the source into
 * a small reused canvas first cuts that upload (and its memory) sharply with no
 * accuracy loss; keypoints are mapped back to source-pixel space afterwards so
 * every downstream consumer (skeleton draw, normalize, composite) is unchanged.
 */
const DETECT_INPUT_MAX = 256;

type ImageSource =
  | HTMLVideoElement
  | HTMLCanvasElement
  | HTMLImageElement;

/** Intrinsic pixel size of a detector input source. */
function sourceSize(src: ImageSource): { w: number; h: number } {
  if (src instanceof HTMLVideoElement) {
    return { w: src.videoWidth, h: src.videoHeight };
  }
  if (src instanceof HTMLImageElement) {
    return { w: src.naturalWidth, h: src.naturalHeight };
  }
  return { w: src.width, h: src.height };
}

type MoveNetModelType =
  (typeof poseDetection.movenet.modelType)[keyof typeof poseDetection.movenet.modelType];

const MOVENET_MODEL_TYPE: Record<"lightning" | "thunder", MoveNetModelType> = {
  lightning: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
  thunder: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
};

function isBlazePose(v: ModelVariant): boolean {
  return v === "blazepose";
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
  // Single reusable canvas the detector input is downscaled into, so we don't
  // allocate a fresh canvas (and its backing memory) every frame.
  private scratch: HTMLCanvasElement | null = null;
  private scratchCtx: CanvasRenderingContext2D | null = null;
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
    // Release WebGL textures as soon as they're freed instead of pooling them.
    // The per-frame detection allocates and discards textures constantly; left
    // pooled, that pool grows and shows up as monotonic memory growth across a
    // clip. A threshold of 0 keeps GPU memory flat (TF.js memory guidance).
    tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 0);
    this.backendReady = true;
  }

  /**
   * Downscale `source` into the reused scratch canvas if it's larger than the
   * detector's tiny input, and return the surface to run inference on plus the
   * factors that map the returned keypoints back into source-pixel space.
   * Returns the source unchanged (identity scale) when it's already small.
   */
  private prepareInput(source: ImageSource): {
    input: ImageSource;
    scaleX: number;
    scaleY: number;
  } {
    const { w, h } = sourceSize(source);
    const longest = Math.max(w, h);
    if (w <= 0 || h <= 0 || longest <= DETECT_INPUT_MAX) {
      return { input: source, scaleX: 1, scaleY: 1 };
    }
    const s = DETECT_INPUT_MAX / longest;
    const dw = Math.max(1, Math.round(w * s));
    const dh = Math.max(1, Math.round(h * s));
    if (!this.scratch) {
      this.scratch = document.createElement("canvas");
      this.scratchCtx = this.scratch.getContext("2d");
    }
    if (this.scratch.width !== dw || this.scratch.height !== dh) {
      this.scratch.width = dw;
      this.scratch.height = dh;
    }
    const ctx = this.scratchCtx;
    if (!ctx) return { input: source, scaleX: 1, scaleY: 1 };
    ctx.drawImage(source, 0, 0, dw, dh);
    return { input: this.scratch, scaleX: w / dw, scaleY: h / dh };
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
    // Downscale the 2D (MoveNet) input to cut the per-frame GPU upload + memory.
    // BlazePose runs its own ROI pipeline on the mediapipe runtime, so it keeps
    // the raw source; its 2D keypoints are untouched there (scale 1).
    const { input, scaleX, scaleY } = this.is3D
      ? { input: source, scaleX: 1, scaleY: 1 }
      : this.prepareInput(source);
    const poses = await this.detector.estimatePoses(input, {
      maxPoses: 1,
      flipHorizontal: false,
    });
    if (!poses.length) return null;
    const p = poses[0];

    const pose: Pose = {
      // Map keypoints from the (possibly downscaled) input back into the
      // source video's pixel space so downstream drawing/scoring is unchanged.
      keypoints: p.keypoints.map((k) => ({
        x: k.x * scaleX,
        y: k.y * scaleY,
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
    const poses = await this.multiDetector!.estimatePoses(source, {
      maxPoses,
      flipHorizontal: false,
    });
    return poses.map(toPose);
  }

  dispose(): void {
    this.detector?.dispose();
    this.detector = null;
    this.multiDetector?.dispose();
    this.multiDetector = null;
  }
}
