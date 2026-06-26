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
