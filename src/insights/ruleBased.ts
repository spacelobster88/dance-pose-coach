// Rule-based coaching provider: the always-available, fully-offline fallback.
//
// It deterministically turns the structured report (#9's report, built by
// report.ts) into prioritized Markdown coaching — no model, no network. This is
// both the graceful fallback when no LLM is configured/reachable and a valid
// provider in its own right, preserving the app's "no upload" promise.

import type { CoachingInput, CoachingProvider, TokenSink } from "./types";

/** Encouraging qualitative band for an average score. */
function band(score: number): string {
  if (score >= 90) return "excellent — you're closely tracking the routine";
  if (score >= 80) return "strong, with a few specific spots to tighten";
  if (score >= 70) return "solid foundation — a handful of limbs need attention";
  if (score >= 55) return "coming together — focus the fixes below";
  return "early days — let's lock in the big shapes first";
}

/** Describe the overall trend from the down-sampled score series. */
function trend(report: CoachingInput): string {
  const s = report.scoreSeries;
  if (s.length < 3) return "";
  const head = s.slice(0, Math.ceil(s.length / 3));
  const tail = s.slice(-Math.ceil(s.length / 3));
  const avg = (xs: { score: number }[]) =>
    xs.reduce((a, b) => a + b.score, 0) / xs.length;
  const delta = avg(tail) - avg(head);
  if (delta > 4) return " You finished stronger than you started — nice build.";
  if (delta < -4) return " You drifted in the back half — stamina or memory may be fading.";
  return " Your accuracy stayed fairly steady throughout.";
}

/** Compose the full Markdown report. */
export function ruleBasedCoaching(report: CoachingInput): string {
  if (report.frames === 0) {
    return "Play a routine first — there's no scored movement to coach yet.";
  }

  const out: string[] = [];
  out.push(
    `**Overall ${report.avgScore}/100** — ${band(report.avgScore)}.${trend(report)}`,
  );

  out.push("");
  out.push("### Top fixes");
  if (report.opportunities.length === 0) {
    out.push(
      "- No single limb stands out — your shapes match well. Push for sharper " +
        "timing and fuller extension to gain the last few points.",
    );
  } else {
    report.opportunities.forEach((o, i) => {
      // Find the segment where this limb was worst, for a time cue.
      let when = "";
      let worstDeg = -1;
      for (const s of report.segments) {
        if (s.worstLimb?.key === o.key && s.worstLimb.degrees > worstDeg) {
          worstDeg = s.worstLimb.degrees;
          when = ` (most off around ${s.startSec}-${s.endSec}s)`;
        }
      }
      out.push(
        `${i + 1}. **${o.label}** is ~${o.degrees}° off${when} — ${o.direction}.`,
      );
    });
  }

  out.push("");
  out.push("### Practice drill");
  const top = report.opportunities[0];
  if (top) {
    out.push(
      `Loop the section where your **${top.label.toLowerCase()}** drifts and run ` +
        "it at half speed, exaggerating the correction — " +
        `${top.direction} — until it feels automatic, then return to tempo.`,
    );
  } else {
    out.push(
      "Run the full routine once more at tempo, focusing on crisp starts and " +
        "stops so each shape lands exactly on the beat.",
    );
  }

  if (report.lowest) {
    out.push("");
    out.push(
      `_Lowest point: ${report.lowest.score}/100 at ${report.lowest.t}s — give ` +
        "that transition an extra rep._",
    );
  }

  return out.join("\n");
}

/** Provider wrapper around the deterministic generator. */
export const ruleBasedProvider: CoachingProvider = {
  id: "rule-based",
  label: "Rule-based (offline)",
  available: async () => true,
  async generate(report: CoachingInput, onToken?: TokenSink): Promise<string> {
    const text = ruleBasedCoaching(report);
    onToken?.(text);
    return text;
  },
};
