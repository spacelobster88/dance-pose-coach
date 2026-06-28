// Rule-based coaching provider: the always-available, fully-offline fallback.
//
// It deterministically turns the structured report (#9's report, built by
// report.ts) into prioritized Markdown coaching — no model, no network. This is
// both the graceful fallback when no LLM is configured/reachable and a valid
// provider in its own right, preserving the app's "no upload" promise.
//
// Output is bilingual to match the LLM providers: a '## English' block followed
// by a '## 中文' block carrying the same advice. The Chinese phrasing is derived
// from the raw correction vector (dx/dy) and limb key, not machine-translated
// from the English string, so it reads naturally to a Chinese-speaking dancer.

import type { CoachingInput, CoachingProvider, LimbCorrection, TokenSink } from "./types";

// Matches report.ts's OFFSET_DEADZONE: below this magnitude a direction
// component is sub-noise and dropped from the Chinese phrase.
const OFFSET_DEADZONE = 0.06;

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
  const d = trendDelta(report);
  if (d === null) return "";
  if (d > 4) return " You finished stronger than you started — nice build.";
  if (d < -4) return " You drifted in the back half — stamina or memory may be fading.";
  return " Your accuracy stayed fairly steady throughout.";
}

/** Mean late-third minus early-third score, or null when too short to judge. */
function trendDelta(report: CoachingInput): number | null {
  const s = report.scoreSeries;
  if (s.length < 3) return null;
  const head = s.slice(0, Math.ceil(s.length / 3));
  const tail = s.slice(-Math.ceil(s.length / 3));
  const avg = (xs: { score: number }[]) =>
    xs.reduce((a, b) => a + b.score, 0) / xs.length;
  return avg(tail) - avg(head);
}

/** When (if ever) a given limb was at its worst, as an English time cue. */
function worstWindow(report: CoachingInput, key: string): { startSec: number; endSec: number } | null {
  let worstDeg = -1;
  let win: { startSec: number; endSec: number } | null = null;
  for (const s of report.segments) {
    if (s.worstLimb?.key === key && s.worstLimb.degrees > worstDeg) {
      worstDeg = s.worstLimb.degrees;
      win = { startSec: s.startSec, endSec: s.endSec };
    }
  }
  return win;
}

// --- English ----------------------------------------------------------------

function buildEnglish(report: CoachingInput): string[] {
  const out: string[] = [];
  out.push("## English");
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
      const w = worstWindow(report, o.key);
      const when = w ? ` (most off around ${w.startSec}-${w.endSec}s)` : "";
      out.push(`${i + 1}. **${o.label}** is ~${o.degrees}° off${when} — ${o.direction}.`);
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
  return out;
}

// --- 中文 (Simplified Chinese) ------------------------------------------------

const ZH_LABELS: Record<string, string> = {
  left_arm: "左臂",
  right_arm: "右臂",
  left_leg: "左腿",
  right_leg: "右腿",
  torso: "躯干",
};

function zhLabel(c: LimbCorrection): string {
  return ZH_LABELS[c.key] ?? c.label;
}

/** Encouraging qualitative band, in Chinese. */
function bandZh(score: number): string {
  if (score >= 90) return "非常出色 —— 你和参考动作高度吻合";
  if (score >= 80) return "很不错，还有几处细节可以收紧";
  if (score >= 70) return "基础扎实 —— 有几个肢体需要注意";
  if (score >= 55) return "渐入佳境 —— 重点打磨下面几项";
  return "刚刚起步 —— 先把大的体态定型";
}

/** Overall trend, in Chinese. */
function trendZh(report: CoachingInput): string {
  const d = trendDelta(report);
  if (d === null) return "";
  if (d > 4) return " 你越跳越稳，结尾比开头更好 —— 状态在往上走。";
  if (d < -4) return " 后半段有所松懈 —— 可能是体力或记忆在下降。";
  return " 全程的准确度保持得比较稳定。";
}

/** Chinese correction phrase from the signed vector (+dx = right, +dy = down). */
function phraseForZh(c: LimbCorrection): string {
  const parts: string[] = [];
  if (c.dy <= -OFFSET_DEADZONE) parts.push("抬高");
  else if (c.dy >= OFFSET_DEADZONE) parts.push("放低");
  if (c.dx <= -OFFSET_DEADZONE) parts.push("向画面左侧带");
  else if (c.dx >= OFFSET_DEADZONE) parts.push("向画面右侧带");
  if (parts.length === 0) return "对齐到参考角度";
  return parts.join("并");
}

function buildChinese(report: CoachingInput): string[] {
  const out: string[] = [];
  out.push("## 中文");
  out.push(`**整体 ${report.avgScore}/100** —— ${bandZh(report.avgScore)}。${trendZh(report)}`);

  out.push("");
  out.push("### 三个重点修正");
  if (report.opportunities.length === 0) {
    out.push(
      "- 没有哪个肢体特别突出 —— 你的体态吻合得不错。在节奏的干净利落和延伸的" +
        "充分到位上再下功夫，把最后几分拿到手。",
    );
  } else {
    report.opportunities.forEach((o, i) => {
      const w = worstWindow(report, o.key);
      const when = w ? `（${w.startSec}-${w.endSec}秒 偏差最大）` : "";
      out.push(`${i + 1}. **${zhLabel(o)}** 约偏差 ${o.degrees}°${when} —— ${phraseForZh(o)}。`);
    });
  }

  out.push("");
  out.push("### 练习方法");
  const top = report.opportunities[0];
  if (top) {
    const w = worstWindow(report, top.key);
    const when = w ? `${w.startSec}-${w.endSec}秒` : "**" + zhLabel(top) + "** 偏差最大的那一段";
    out.push(
      `把 ${when} 这段循环播放，放慢到一半速度，刻意把修正做大 —— ` +
        `${phraseForZh(top)} —— 直到形成肌肉记忆，再回到原速。`,
    );
  } else {
    out.push("再完整跑一遍，注意每个动作的起势与收势干净利落，让每个体态都精准卡在节拍上。");
  }

  if (report.lowest) {
    out.push("");
    out.push(`_最低点：${report.lowest.score}/100，出现在 ${report.lowest.t}秒 —— 这个衔接多练几遍。_`);
  }
  return out;
}

/** Compose the full bilingual Markdown report. */
export function ruleBasedCoaching(report: CoachingInput): string {
  if (report.frames === 0) {
    return (
      "Play a routine first — there's no scored movement to coach yet.\n\n" +
      "先放一段舞蹈 —— 目前还没有可评分的动作可供指导。"
    );
  }

  return [...buildEnglish(report), "", ...buildChinese(report)].join("\n");
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
