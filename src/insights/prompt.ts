// Prompt construction shared by the LLM providers (Ollama / Claude proxy).
//
// The system prompt frames the model as a dance coach; the user message is the
// compact structured report rendered as terse, token-cheap text. We deliberately
// send only derived stats (degrees, signed nudges, scores) — never frames.

import type { CoachingInput } from "./types";

export const SYSTEM_PROMPT =
  "You are an expert, encouraging dance coach. You are given a per-segment " +
  "breakdown of how closely a student matched a reference routine: each limb's " +
  "angular error in degrees (0 = perfect, higher = more off) and a signed " +
  "direction telling the student which way to move that limb to match. " +
  "Produce concise Markdown with exactly these parts:\n" +
  "1. A one-line overall read (mention the average score and the trend).\n" +
  "2. '### Top 3 fixes' — three prioritized, specific, encouraging corrections, " +
  "each naming the limb, the size of the error, and the concrete direction to " +
  "move. Reference the time in the routine when useful.\n" +
  "3. '### Practice drill' — one short, concrete drill targeting the biggest " +
  "issue.\n" +
  "Be specific and positive. Do not invent data beyond what is provided. Keep it " +
  "under ~200 words.";

/** Render the report as compact, deterministic text for the user turn. */
export function buildUserMessage(report: CoachingInput): string {
  const lines: string[] = [];
  lines.push(
    `Routine length: ${report.durationSec}s over ${report.frames} analysed frames.`,
  );
  lines.push(`Average similarity: ${report.avgScore}/100.`);
  if (report.lowest) {
    lines.push(`Lowest moment: ${report.lowest.score}/100 at ${report.lowest.t}s.`);
  }

  if (report.opportunities.length) {
    lines.push("");
    lines.push("Biggest overall opportunities (worst first):");
    for (const o of report.opportunities) {
      lines.push(`- ${o.label}: ${o.degrees}° off — ${o.direction}.`);
    }
  }

  if (report.segments.length) {
    lines.push("");
    lines.push("Per-segment worst limb:");
    for (const s of report.segments) {
      const w = s.worstLimb
        ? `${s.worstLimb.label} ${s.worstLimb.degrees}° (${s.worstLimb.direction})`
        : "no clear weak limb";
      lines.push(
        `- ${s.startSec}-${s.endSec}s: avg ${s.avgScore}/100, worst ${w}.`,
      );
    }
  }

  if (report.scoreSeries.length) {
    const series = report.scoreSeries.map((p) => p.score).join(", ");
    lines.push("");
    lines.push(`Score timeline: ${series}.`);
  }

  return lines.join("\n");
}
