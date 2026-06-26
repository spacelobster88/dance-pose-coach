// Coaching-insights entry point: a small registry of model-agnostic providers
// plus a dispatcher that picks one and falls back cleanly to the offline
// rule-based report when nothing else is configured or a provider fails.
//
// Public surface:
//   - PROVIDERS / getProvider(): the registry
//   - generateCoaching(report, opts): run a provider, with graceful fallback
//   - RunReport, frameLimbMetrics: report aggregation (re-exported)

import type {
  CoachingInput,
  CoachingProvider,
  TokenSink,
} from "./types";
import { ruleBasedProvider, ruleBasedCoaching } from "./ruleBased";
import { ollamaProvider } from "./ollama";
import { claudeProvider } from "./claude";

export type { CoachingInput, CoachingProvider, TokenSink } from "./types";
export { RunReport, frameLimbMetrics } from "./report";
export { renderMarkdown } from "./markdown";
export { ruleBasedCoaching } from "./ruleBased";

/** "auto" lets the dispatcher choose the best available provider. */
export type ProviderChoice = "auto" | "ollama" | "claude" | "rule-based";

/** All registered providers, keyed by id. */
export const PROVIDERS: Record<string, CoachingProvider> = {
  [ollamaProvider.id]: ollamaProvider,
  [claudeProvider.id]: claudeProvider,
  [ruleBasedProvider.id]: ruleBasedProvider,
};

export function getProvider(id: string): CoachingProvider | undefined {
  return PROVIDERS[id];
}

export interface GenerateOptions {
  /** Which provider to use; "auto" probes for the best available. */
  provider?: ProviderChoice;
  /** Streaming sink for incremental output. */
  onToken?: TokenSink;
}

export interface CoachingResult {
  /** The generated coaching, as Markdown. */
  text: string;
  /** Which provider actually produced the text. */
  providerId: string;
  /** True when we fell back to the rule-based provider. */
  fellBack: boolean;
  /** Set when a chosen provider was unavailable or errored. */
  note?: string;
}

/** Resolve "auto" to the first available LLM provider, else rule-based. */
async function resolveAuto(): Promise<CoachingProvider> {
  if (await ollamaProvider.available()) return ollamaProvider;
  if (await claudeProvider.available()) return claudeProvider;
  return ruleBasedProvider;
}

/**
 * Generate coaching for a run report. Selects a provider per `opts.provider`,
 * streams via `opts.onToken`, and on any unavailability/failure falls back to
 * the always-on rule-based report so the action never dead-ends.
 */
export async function generateCoaching(
  report: CoachingInput,
  opts: GenerateOptions = {},
): Promise<CoachingResult> {
  const choice = opts.provider ?? "auto";

  let provider: CoachingProvider;
  let note: string | undefined;

  if (choice === "auto") {
    provider = await resolveAuto();
  } else {
    const chosen = PROVIDERS[choice];
    if (chosen && (await chosen.available())) {
      provider = chosen;
    } else {
      // Chosen provider isn't usable — fall back, and say why.
      note =
        choice === "ollama"
          ? "Ollama isn't reachable on localhost:11434 — used the offline report instead."
          : choice === "claude"
            ? "No Claude proxy is configured — used the offline report instead."
            : undefined;
      provider = ruleBasedProvider;
    }
  }

  // Rule-based can't fail; for it, just generate.
  if (provider.id === ruleBasedProvider.id) {
    const text = await provider.generate(report, opts.onToken);
    return { text, providerId: provider.id, fellBack: choice !== "rule-based", note };
  }

  try {
    const text = await provider.generate(report, opts.onToken);
    return { text, providerId: provider.id, fellBack: false };
  } catch (err) {
    // Hard failure mid-generation: fall back so the user still gets coaching.
    const text = ruleBasedCoaching(report);
    opts.onToken?.(text);
    return {
      text,
      providerId: ruleBasedProvider.id,
      fellBack: true,
      note: `${provider.label} failed (${
        err instanceof Error ? err.message : "unknown error"
      }) — used the offline report instead.`,
    };
  }
}
