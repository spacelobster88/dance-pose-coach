import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

import type { Pose } from "./keypoints";

export type ModelVariant = "lightning" | "thunder";

type MoveNetModelType =
  (typeof poseDetection.movenet.modelType)[keyof typeof poseDetection.movenet.modelType];

const MODEL_TYPE: Record<ModelVariant, MoveNetModelType> = {
  lightning: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
  thunder: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
};

/**
 * Thin wrapper around MoveNet single-pose detection.
 *
 * Holds one detector instance and lets the UI hot-swap between the Lightning
 * (fast) and Thunder (accurate) variants without tearing down the whole app.
 */
export class PoseDetector {
  private detector: poseDetection.PoseDetector | null = null;
  private variant: ModelVariant;
  private loadingPromise: Promise<void> | null = null;
  private backendReady = false;

  constructor(variant: ModelVariant = "lightning") {
    this.variant = variant;
  }

  get currentVariant(): ModelVariant {
    return this.variant;
  }

  /** Ensure the WebGL backend is selected and ready exactly once. */
  private async ensureBackend(): Promise<void> {
    if (this.backendReady) return;
    await tf.setBackend("webgl");
    await tf.ready();
    this.backendReady = true;
  }

  /** Load (or reload) the MoveNet model for the current variant. */
  async load(): Promise<void> {
    if (this.loadingPromise) return this.loadingPromise;
    this.loadingPromise = (async () => {
      await this.ensureBackend();
      this.detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: MODEL_TYPE[this.variant],
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
   * Returns null when no person is detected.
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
    const p = poses[0];
    return {
      keypoints: p.keypoints.map((k) => ({
        x: k.x,
        y: k.y,
        score: k.score ?? undefined,
        name: k.name,
      })),
      score: p.score ?? undefined,
    };
  }

  dispose(): void {
    this.detector?.dispose();
    this.detector = null;
  }
}
