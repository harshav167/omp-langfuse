import { normalizeAnthropicGoogle } from "./anthropic-google.ts";
import { normalizeOpenAI } from "./openai.ts";
import type { NormalizedRequest } from "./types.ts";
import type { TracingHistoryMessage } from "../index.ts";


/** Project the exact provider payload, for both live tracing and archive replay. */
export function normalizeProviderRequest(payload: unknown, api: string): NormalizedRequest {
  switch (api) {
    case "openai-completions":
    case "openai-responses":
    case "openai-codex-responses":
      return normalizeOpenAI(payload, api);
    case "anthropic-messages":
    case "google-generative-ai":
    case "google-vertex":
    case "google-gemini-cli":
      return normalizeAnthropicGoogle(payload, api);
    default:
      throw new Error(`Unsupported provider API ${api}`);
  }
}
export const RESPONSES_APIS: ReadonlySet<string> = new Set(["openai-responses", "openai-codex-responses"]);
export interface ToolCallCandidate {
  id: string;
  name: string;
  api: string;
}
export function callAlias(id: string, api: string): string {
  if (RESPONSES_APIS.has(api)) {
    const separator = id.indexOf("|");
    if (separator > 0) return id.slice(0, separator);
  }
  return id;
}
function assistantToolCandidates(messages: readonly TracingHistoryMessage[], api: string): ToolCallCandidate[] {
  const candidates: ToolCallCandidate[] = [];
  for (const message of messages) {
    if (message.role !== "assistant" || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (part.type !== "toolCall") continue;
      candidates.push({ id: part.id, name: part.name, api });
    }
  }
  return candidates;
}
function matchesToolCall(
  result: TracingHistoryMessage,
  candidates: readonly ToolCallCandidate[],
  resultApi: string,
): ToolCallCandidate[] {
  if (result.role !== "toolResult" || typeof result.toolCallId !== "string") return [];
  const resultId = result.toolCallId;
  return candidates.filter(candidate =>
    candidate.id === resultId ||
    (RESPONSES_APIS.has(candidate.api) && RESPONSES_APIS.has(resultApi) &&
      callAlias(candidate.id, candidate.api) === callAlias(resultId, resultApi)),
  );
}
export function enrichToolResults(
  messages: TracingHistoryMessage[],
  api: string,
  pending: Iterable<ToolCallCandidate> = [],
): { messages: TracingHistoryMessage[]; missingNames: number; ambiguousNames: number } {
  const candidates = [
    ...assistantToolCandidates(messages, api),
    ...pending,
  ];
  let missingNames = 0;
  let ambiguousNames = 0;
  let enriched = messages;
  for (const [index, message] of messages.entries()) {
    if (message.role !== "toolResult") continue;
    if (typeof message.toolName === "string" && message.toolName.length > 0) continue;
    const matches = matchesToolCall(message, candidates, api);
    const names = [...new Set(matches.map(candidate => candidate.name).filter(name => name.length > 0))];
    // Duplicate record forms can identify the same name without identifying one
    // execution. Naming is safe when every match agrees; execution stays separately matched.
    const toolName = names.length === 1 ? names[0] : undefined;
    if (toolName === undefined) {
      missingNames++;
      if (matches.length > 1) ambiguousNames++;
    }
    if (toolName !== message.toolName) {
      if (enriched === messages) enriched = messages.slice();
      const { toolName: _unknownName, ...result } = message;
      enriched[index] = toolName === undefined ? result : { ...result, toolName };
    }
  }
  return { messages: enriched, missingNames, ambiguousNames };
}
