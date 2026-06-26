// Remote provider: Claude via a thin, user-supplied proxy.
//
// Best quality, but it sends *derived* stats off-device, so it is strictly
// opt-in — it is only "available" once a proxy URL is configured. The API key
// lives in the proxy (a small serverless function), never in client code. The
// proxy receives the structured report + system prompt and must never see video.
//
// Expected proxy contract (kept deliberately simple and model-agnostic):
//   POST <proxyUrl>  { system, message, report, model }
//   → either text/event-stream of Anthropic-style deltas
//     (data: {"type":"content_block_delta","delta":{"text":"..."}})
//   → or a JSON body { text: "..." }
//   → or a plain text/markdown body.

import type { CoachingInput, CoachingProvider, TokenSink } from "./types";
import { SYSTEM_PROMPT, buildUserMessage } from "./prompt";
import { getLocal } from "./storage";

/** Default remote model — latest Claude per the issue recommendation. */
const DEFAULT_MODEL = "claude-opus-4-8";

function proxyUrl(): string {
  return (getLocal("dpc.coach.claude.proxyUrl") || "").trim();
}

function model(): string {
  return getLocal("dpc.coach.claude.model") || DEFAULT_MODEL;
}

/** Parse one SSE `data:` payload, returning any text delta it carries. */
function deltaFromSse(payload: string): string {
  try {
    const obj = JSON.parse(payload);
    if (obj?.type === "content_block_delta" && typeof obj?.delta?.text === "string") {
      return obj.delta.text;
    }
    if (typeof obj?.text === "string") return obj.text;
    return "";
  } catch {
    return "";
  }
}

export const claudeProvider: CoachingProvider = {
  id: "claude",
  label: "Remote · Claude",

  async available(): Promise<boolean> {
    // Opt-in: only available once the user has configured a proxy endpoint.
    return proxyUrl().length > 0;
  },

  async generate(report: CoachingInput, onToken?: TokenSink): Promise<string> {
    const url = proxyUrl();
    if (!url) throw new Error("No Claude proxy configured");

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system: SYSTEM_PROMPT,
        message: buildUserMessage(report),
        report,
        model: model(),
      }),
    });
    if (!res.ok) throw new Error(`Claude proxy failed (${res.status})`);

    const ctype = res.headers.get("content-type") || "";

    // Streaming SSE path.
    if (res.body && ctype.includes("text/event-stream")) {
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
          const piece = deltaFromSse(payload);
          if (piece) {
            full += piece;
            onToken?.(piece);
          }
        }
      }
      if (full.trim()) return full;
      // Fall through to non-streaming handling if nothing was parsed.
    }

    // Non-streaming: JSON { text } or raw text/markdown.
    if (ctype.includes("application/json")) {
      const data = await res.json();
      const text: string =
        typeof data?.text === "string"
          ? data.text
          : typeof data?.content === "string"
            ? data.content
            : "";
      if (!text.trim()) throw new Error("Claude proxy returned no text");
      onToken?.(text);
      return text;
    }

    const text = await res.text();
    if (!text.trim()) throw new Error("Claude proxy returned an empty response");
    onToken?.(text);
    return text;
  },
};
