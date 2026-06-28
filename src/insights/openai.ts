// Remote provider: OpenAI Chat Completions, called directly with a user-supplied
// API key.
//
// Best quality without standing up a proxy: the dancer pastes their own key, which
// is stored only in this browser's localStorage and sent straight to OpenAI over
// HTTPS. Like the Claude path it is strictly opt-in and sends only *derived* stats
// (degrees, signed nudges, scores) — never video and never raw frames. For users
// who would rather keep the key off the client, an optional base-URL override lets
// them point at an OpenAI-compatible proxy (or Azure) instead.
//
// Security note: a client-side key is visible to scripts on this origin. That is an
// accepted tradeoff for a local, no-server tool; the key never leaves the browser
// except in the request to the configured endpoint, and is never logged.

import type { CoachingInput, CoachingProvider, TokenSink } from "./types";
import { SYSTEM_PROMPT, buildUserMessage } from "./prompt";
import { getLocal } from "./storage";

export interface OpenAIConfig {
  /** Secret API key (browser-only). */
  apiKey: string;
  /** Chat model id, e.g. "gpt-4o-mini". */
  model: string;
  /** API base URL — override for Azure / OpenAI-compatible / proxy endpoints. */
  baseUrl: string;
}

const DEFAULTS = {
  model: "gpt-4o-mini",
  baseUrl: "https://api.openai.com/v1",
};

// Build-time key, injected by Vite's `define` from the DANCE_COACH_OPENAI_KEY
// shell var (see vite.config.ts). Replaced with a string literal at bundle time;
// outside Vite (unit tests / plain Node) the identifier is undeclared, so the
// `typeof` guard yields "" instead of throwing — keeping OpenAI opt-in there.
declare const __DPC_OPENAI_KEY__: string;

function buildTimeKey(): string {
  try {
    return typeof __DPC_OPENAI_KEY__ === "string" ? __DPC_OPENAI_KEY__.trim() : "";
  } catch {
    return "";
  }
}

function readConfig(): OpenAIConfig {
  return {
    // A key pasted into the panel (localStorage) wins; otherwise fall back to the
    // one baked in at build time. Both stay browser-only.
    apiKey: (getLocal("dpc.coach.openai.apiKey") || buildTimeKey() || "").trim(),
    model: getLocal("dpc.coach.openai.model") || DEFAULTS.model,
    // Trim any trailing slash so we can append paths predictably.
    baseUrl: (getLocal("dpc.coach.openai.baseUrl") || DEFAULTS.baseUrl).replace(/\/+$/, ""),
  };
}

/** Pull the text delta out of one streamed chat-completion chunk. */
function deltaFromChunk(payload: string): string {
  try {
    const obj = JSON.parse(payload);
    const piece = obj?.choices?.[0]?.delta?.content;
    return typeof piece === "string" ? piece : "";
  } catch {
    return "";
  }
}

export const openaiProvider: CoachingProvider = {
  id: "openai",
  label: "Remote · OpenAI",

  async available(): Promise<boolean> {
    // Opt-in: only usable once the user has pasted an API key. No network probe —
    // a bad key surfaces as a clear error at generate() time instead.
    return readConfig().apiKey.length > 0;
  },

  async generate(report: CoachingInput, onToken?: TokenSink): Promise<string> {
    const { apiKey, model, baseUrl } = readConfig();
    if (!apiKey) throw new Error("No OpenAI API key configured");

    const res = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        stream: true,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: buildUserMessage(report) },
        ],
      }),
    });
    if (!res.ok || !res.body) {
      // Surface the status (401 bad key, 404 unknown model, 429 rate-limited, …)
      // so the dispatcher's fallback note tells the user what to fix.
      throw new Error(`OpenAI request failed (${res.status})`);
    }

    // Chat Completions streams Server-Sent Events: lines of `data: {json}`, each
    // carrying choices[0].delta.content, terminated by `data: [DONE]`.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let full = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        const piece = deltaFromChunk(payload);
        if (piece) {
          full += piece;
          onToken?.(piece);
        }
      }
    }
    if (!full.trim()) throw new Error("OpenAI returned an empty response");
    return full;
  },
};
