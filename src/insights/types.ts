// Shared types for the coaching-insights layer.
//
// The LLM (or the rule-based fallback) is fed a *compact, derived* structured
// report — never video and never raw frames. Only pose-derived statistics
// (per-limb angular errors in degrees, signed correction directions, a
// down-sampled score timeline, and the top correction opportunities) leave the
// page, which keeps the app's "no upload" privacy promise intact for the local
// provider and minimizes what an opt-in remote provider ever sees.

/** A single down-sampled point on the similarity timeline. */
export interface ScorePoint {
  /** Seconds into the routine. */
  t: number;
  /** Similarity score in [0, 100] at this instant. */
  score: number;
}

/**
 * Signed correction for one limb, expressed as the direction the dancer should
 * move to match the reference (i.e. the negated test−ref offset), in
 * torso-normalized units. `dx > 0` ⇒ move toward screen-right, `dy > 0` ⇒ move
 * down. A short human phrase is precomputed for the rule-based path and as a
 * hint for the LLM.
 */
export interface LimbCorrection {
  /** Limb group key (matches perJoint.LIMB_GROUPS). */
  key: string;
  /** Human label, e.g. "Left arm". */
  label: string;
  /** Mean bone-direction error for this limb, in degrees [0, 180]. */
  degrees: number;
  /** Signed horizontal correction (torso units); + = move screen-right. */
  dx: number;
  /** Signed vertical correction (torso units); + = move down. */
  dy: number;
  /** Precomputed natural-language nudge, e.g. "lift and move screen-left". */
  direction: string;
}

/** One contiguous time slice of the routine, summarized. */
export interface ReportSegment {
  index: number;
  startSec: number;
  endSec: number;
  /** Mean similarity score over the segment, [0, 100]. */
  avgScore: number;
  /** The worst-diverging limb in this segment (null if no data). */
  worstLimb: LimbCorrection | null;
}

/** The compact structured report handed to a coaching provider. */
export interface CoachingInput {
  /** Total analysed duration in seconds. */
  durationSec: number;
  /** Number of scored frames the report is built from. */
  frames: number;
  /** Mean similarity over the whole run, [0, 100]. */
  avgScore: number;
  /** Lowest scored instant and when it happened. */
  lowest: { score: number; t: number } | null;
  /** Time-ordered segment summaries. */
  segments: ReportSegment[];
  /** Top correction opportunities across the run, worst first. */
  opportunities: LimbCorrection[];
  /** Down-sampled score timeline for trend/shape. */
  scoreSeries: ScorePoint[];
}

/** Streaming sink: called with each new chunk of generated text. */
export type TokenSink = (delta: string) => void;

/** A pluggable, model-agnostic coaching backend. */
export interface CoachingProvider {
  /** Stable identifier, e.g. "ollama". */
  id: string;
  /** Human label for the UI. */
  label: string;
  /**
   * Whether this provider can run right now (endpoint reachable / configured).
   * May probe the network; should resolve quickly and never throw.
   */
  available(): Promise<boolean>;
  /**
   * Turn the structured report into natural-language coaching. Implementations
   * should call `onToken` as text streams in (when supported) and also return
   * the full text. Must reject on hard failure so the caller can fall back.
   */
  generate(report: CoachingInput, onToken?: TokenSink): Promise<string>;
}
