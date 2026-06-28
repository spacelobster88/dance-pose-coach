// Local LLM provider via Ollama (http://localhost:11434).
//
// Recommended default: it keeps the no-server/privacy promise (the report never
// leaves the machine) and Eddie already runs Ollama for embeddings. We POST the
// compact report to /api/chat with streaming enabled and surface tokens as they
// arrive.

import type { CoachingInput, CoachingProvider, TokenSink } from "./types";
import { SYSTEM_PROMPT, buildUserMessage } from "./prompt";
import { getLocal } from "./storage";

export interface OllamaConfig {
  /** Base URL of the Ollama server. */
  endpoint: string;
  /** Model tag to use, e.g. "llama3.1". */
  model: string;
}

const DEFAULTS: OllamaConfig = {
  endpoint: "http://localhost:11434",
  model: "llama3.1",
};

function readConfig(): OllamaConfig {
  return {
    endpoint: getLocal("dpc.coach.ollama.endpoint") || DEFAULTS.endpoint,
    model: getLocal("dpc.coach.ollama.model") || DEFAULTS.model,
  };
}

/** Short-timeout fetch helper so availability probes never hang the UI. */
async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

export const ollamaProvider: CoachingProvider = {
  id: "ollama",
  label: "Local · Ollama",

  async available(): Promise<boolean> {
    const { endpoint } = readConfig();
    try {
      const res = await fetchWithTimeout(`${endpoint}/api/tags`, { method: "GET" }, 1500);
      return res.ok;
    } catch {
      return false;
    }
  },

  async generate(report: CoachingInput, onToken?: TokenSink): Promise<string> {
    const { endpoint, model } = readConfig();
    const res = await fetch(`${endpoint}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
      // A 404 from /api/chat almost always means the model tag isn't pulled
      // (the /api/tags availability probe still passes when the server is up).
      // Point the user at the fix instead of a bare status code.
      if (res.status === 404) {
        throw new Error(`Ollama model "${model}" not found — run: ollama pull ${model}`);
      }
      throw new Error(`Ollama request failed (${res.status})`);
    }

    // /api/chat streams newline-delimited JSON; each line carries a partial
    // message.content until { done: true }.
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
        if (!line) continue;
        try {
          const obj = JSON.parse(line);
          const piece: string | undefined = obj?.message?.content;
          if (piece) {
            full += piece;
            onToken?.(piece);
          }
        } catch {
          // Ignore partial/non-JSON lines; the next chunk completes them.
        }
      }
    }
    if (!full.trim()) throw new Error("Ollama returned an empty response");
    return full;
  },
};
