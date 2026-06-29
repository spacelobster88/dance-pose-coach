// Bilingual "these are two different dances" message.
//
// When the report's mismatch assessment fires (see assessMismatch in report.ts),
// the dispatcher returns this instead of per-limb coaching: the similarity score
// isn't a meaningful measure when the two clips aren't the same choreography, so
// telling the dancer to "lift the torso" would be nonsensical. English + 简体中文,
// matching the bilingual coaching format.

import type { CoachingInput } from "./types";

/** Bilingual Markdown explaining the clips look like different dances. */
export function mismatchMessage(report: CoachingInput): string {
  const avg = report.avgScore;
  const peak = report.peakScore;
  return [
    "## English",
    "",
    "⚠️ **These look like two different dances.** The similarity score " +
      `(${avg}/100, best moment ${peak}/100) isn't a meaningful measure of ` +
      "performance — the two clips never line up, so per-move coaching wouldn't " +
      "make sense.",
    "",
    "Pick a reference and an attempt of the **same** routine to get coaching.",
    "",
    "## 中文",
    "",
    "⚠️ **这看起来是两段不同的舞蹈。** 相似度分数" +
      `（${avg}/100，最高瞬间 ${peak}/100）` +
      "无法作为练习参考——两段视频始终对不上，逐动作的指导没有意义。",
    "",
    "请选择**同一支舞蹈**的参考视频与练习视频，才能获得指导。",
  ].join("\n");
}
