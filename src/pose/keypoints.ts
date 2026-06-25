// COCO-17 keypoint topology shared across the detection, normalization,
// scoring and rendering modules.

/** A single detected keypoint in pixel coordinates of the source video. */
export interface Keypoint {
  x: number;
  y: number;
  /**
   * Optional depth channel. For 2D detectors (MoveNet) this is undefined and
   * every existing consumer ignores it. For 3D detectors (BlazePose) it carries
   * the model's relative depth so 2D-only code keeps behaving identically.
   */
  z?: number;
  /** Detector confidence in [0, 1]; may be undefined for some backends. */
  score?: number;
  name?: string;
}

/** One detected person: an ordered array of the 17 COCO keypoints. */
export interface Pose {
  keypoints: Keypoint[];
  /** Overall pose confidence in [0, 1] when provided by the model. */
  score?: number;
  /**
   * Optional 3D world landmarks in metric space (origin at the hip center),
   * populated only by 3D detectors such as BlazePose. Same index order as
   * `keypoints`. Undefined for MoveNet; 2D code paths never read it.
   */
  worldKeypoints?: Keypoint[];
}

/**
 * COCO-17 keypoint names in the exact index order MoveNet returns them.
 * Index in this array == keypoint index in a Pose.
 */
export const KEYPOINT_NAMES = [
  "nose", // 0
  "left_eye", // 1
  "right_eye", // 2
  "left_ear", // 3
  "right_ear", // 4
  "left_shoulder", // 5
  "right_shoulder", // 6
  "left_elbow", // 7
  "right_elbow", // 8
  "left_wrist", // 9
  "right_wrist", // 10
  "left_hip", // 11
  "right_hip", // 12
  "left_knee", // 13
  "right_knee", // 14
  "left_ankle", // 15
  "right_ankle", // 16
] as const;

export type KeypointName = (typeof KEYPOINT_NAMES)[number];

/** Reverse lookup: keypoint name -> index. */
export const KEYPOINT_INDEX: Record<KeypointName, number> = KEYPOINT_NAMES.reduce(
  (acc, name, i) => {
    acc[name] = i;
    return acc;
  },
  {} as Record<KeypointName, number>,
);

/**
 * Skeleton edges (pairs of keypoint indices) used to draw bones.
 * Mirrors the standard COCO body topology.
 */
export const SKELETON_EDGES: Array<[number, number]> = [
  // Face
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  // Shoulders / torso
  [5, 6],
  [5, 11],
  [6, 12],
  [11, 12],
  // Left arm
  [5, 7],
  [7, 9],
  // Right arm
  [6, 8],
  [8, 10],
  // Left leg
  [11, 13],
  [13, 15],
  // Right leg
  [12, 14],
  [14, 16],
];

/**
 * Keypoints that carry real pose information for scoring. We exclude the face
 * (eyes/ears/nose) because tiny head movements dominate cosine similarity
 * without reflecting dance-relevant body posture.
 */
export const SCORING_KEYPOINTS: KeypointName[] = [
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
];

export const SCORING_KEYPOINT_INDICES: number[] = SCORING_KEYPOINTS.map(
  (n) => KEYPOINT_INDEX[n],
);
