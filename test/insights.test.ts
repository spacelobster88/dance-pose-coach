// Unit tests for the coaching-insights layer (#15): report aggregation,
// rule-based provider, Markdown rendering, and the dispatcher's graceful
// fallback. Pure logic only — no DOM, no network reachable in CI, so the
// "auto" provider must resolve to the offline rule-based report.
//
//   npm test
//
// Exits non-zero on any failed assertion.

import { RunReport, frameLimbMetrics } from "../src/insights/report.ts";
import { ruleBasedCoaching } from "../src/insights/ruleBased.ts";
import { renderMarkdown } from "../src/insights/markdown.ts";
import { generateCoaching } from "../src/insights/index.ts";
import type { NormalizedPose } from "../src/pose/normalize.ts";

let failures = 0;
const check = (cond: unknown, label: string, detail?: string) => {
  if (cond) {
    console.log(`PASS · ${label}${detail ? ` (${detail})` : ""}`);
  } else {
    console.log(`FAIL · ${label}${detail ? ` (${detail})` : ""}`);
    failures++;
  }
};

// --- Build two synthetic normalized poses (hip-centered, torso ≈ 1) -----------
// Identical except the left arm is raised in `test`, which should surface as the
// top opportunity with a "lower" correction (test arm is above the reference).
function pose(
  overrides: Record<number, [number, number]> = {},
): NormalizedPose {
  const base: Record<number, [number, number]> = {
    0: [0, -0.6], // nose
    1: [-0.05, -0.62],
    2: [0.05, -0.62],
    3: [-0.1, -0.6],
    4: [0.1, -0.6],
    5: [-0.2, -0.5], // left_shoulder
    6: [0.2, -0.5], // right_shoulder
    7: [-0.4, -0.2], // left_elbow
    8: [0.4, -0.2], // right_elbow
    9: [-0.5, 0.1], // left_wrist
    10: [0.5, 0.1], // right_wrist
    11: [-0.15, 0.0], // left_hip
    12: [0.15, 0.0], // right_hip
    13: [-0.15, 0.5], // left_knee
    14: [0.15, 0.5], // right_knee
    15: [-0.15, 1.0], // left_ankle
    16: [0.15, 1.0], // right_ankle
  };
  const points = Array.from({ length: 17 }, (_, i) => {
    const [x, y] = overrides[i] ?? base[i];
    return { x, y, valid: true };
  });
  return { points };
}

const ref = pose();
// Left elbow + wrist raised well above the reference.
const test = pose({ 7: [-0.4, -0.6], 9: [-0.5, -0.9] });

// --- Report aggregation -------------------------------------------------------
const report = new RunReport();
const FRAMES = 30;
for (let i = 0; i < FRAMES; i++) {
  // Score sags in the middle so the timeline has shape.
  const score = i > 10 && i < 20 ? 60 : 85;
  report.push(i * 100, score, frameLimbMetrics(ref, test));
}
check(report.size === FRAMES, "report records every frame", `${report.size}`);

const input = report.build();
check(input.frames === FRAMES, "build keeps frame count");
check(input.segments.length > 1, "timeline is segmented", `${input.segments.length}`);
check(
  input.scoreSeries.length > 0 && input.scoreSeries.length <= 24,
  "score series is down-sampled",
  `${input.scoreSeries.length}`,
);
check(input.lowest !== null && input.lowest.score === 60, "lowest moment captured");

const top = input.opportunities[0];
check(top !== undefined, "an opportunity is surfaced");
check(top?.key === "left_arm", "worst limb is the left arm", top?.key);
check((top?.degrees ?? 0) > 30, "left-arm angular error is large", `${top?.degrees}°`);
check(/lower/.test(top?.direction ?? ""), "correction says to lower the arm", top?.direction);

// A perfectly-matched run yields no opportunities but still builds.
const clean = new RunReport();
for (let i = 0; i < 10; i++) clean.push(i * 100, 99, frameLimbMetrics(ref, ref));
const cleanInput = clean.build();
check(cleanInput.opportunities.length === 0, "identical poses ⇒ no opportunities");

// --- Rule-based provider ------------------------------------------------------
const coaching = ruleBasedCoaching(input);
check(/Left arm/.test(coaching), "rule-based names the left arm");
check(/### Practice drill/.test(coaching), "rule-based includes a practice drill");
check(/Overall [\d.]+\/100/.test(coaching), "rule-based leads with the overall score");
check(
  ruleBasedCoaching(new RunReport().build()).includes("Play a routine"),
  "empty report ⇒ friendly prompt",
);

// --- Markdown rendering -------------------------------------------------------
const html = renderMarkdown(coaching);
check(/<h3>/.test(html), "markdown renders headings");
check(/<strong>/.test(html), "markdown renders bold");
check(/<(ol|ul)>/.test(html), "markdown renders a list");
check(!/<script/i.test(renderMarkdown("<script>alert(1)</script>")), "markdown escapes HTML");

// --- Dispatcher fallback ------------------------------------------------------
const result = await generateCoaching(input, { provider: "auto" });
check(result.text.length > 0, "auto dispatch returns coaching");
check(
  result.providerId === "rule-based",
  "auto falls back to rule-based with no LLM",
  result.providerId,
);

// Explicitly choosing the unreachable Claude provider must fall back with a note.
const remote = await generateCoaching(input, { provider: "claude" });
check(remote.providerId === "rule-based", "unconfigured Claude ⇒ fallback");
check(remote.fellBack === true, "fallback flag set");
check(Boolean(remote.note), "fallback explains why", remote.note);

// Same contract for OpenAI: with no API key configured (no localStorage in CI),
// the remote provider is unavailable and we fall back with an explanatory note.
const openai = await generateCoaching(input, { provider: "openai" });
check(openai.providerId === "rule-based", "unconfigured OpenAI ⇒ fallback");
check(openai.fellBack === true, "OpenAI fallback flag set");
check(/OpenAI/.test(openai.note ?? ""), "OpenAI fallback explains why", openai.note);

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
