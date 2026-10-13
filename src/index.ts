import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAgentIdentity, ExtensionEvent } from "@oh-my-pi/pi-coding-agent";
import type { LangfuseObservation, LangfuseObservationAttributes } from "@langfuse/tracing";
import type { IdGenerator } from "@opentelemetry/sdk-trace-node";
import {
 type ExtensionAPI,
 type ExtensionContext,
 convertToLlm,
 getAgentDir,
} from "@oh-my-pi/pi-coding-agent";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import {
 type LangfuseAgent,
 type LangfuseGeneration,
 LangfuseOtelSpanAttributes,
 type LangfuseRetriever,
 type LangfuseTool,
 setLangfuseTracerProvider,
 startObservation,
} from "@langfuse/tracing";
import { type SpanContext, TraceFlags } from "@opentelemetry/api";
import { defaultResource, detectResources, envDetector, resourceFromAttributes } from "@opentelemetry/resources";
import { AlwaysOnSampler, NodeTracerProvider, RandomIdGenerator } from "@opentelemetry/sdk-trace-node";
import { registerRawCapture } from "./raw-capture.ts";
import { digest } from "./raw-capture-core.ts";
import { enrichToolResults, normalizeProviderRequest } from "./backfill/normalize.ts";
import { registerSdkCompactionUsage } from "./sdk-compaction.ts";
import {
 cacheAssessmentFor,
 cacheMetadata,
 modelLabel,
 modelSwitchFor,
 registerCacheAccounting,
 sessionTitleOf,
} from "./cache-accounting.ts";

const EXTENSION_NAME = "@langfuse/pi-observability-plugin";
const EXTENSION_VERSION = "0.2.0";
// Observation and trace names are an API for evaluators, dashboards and saved
// views: verb-first, low-cardinality, never the model or a per-run value.
const TRACE_NAME = "omp-turn";
const MAIN_AGENT_OBSERVATION_NAME = "omp-agent";
const GENERATION_OBSERVATION_NAME = "call-llm";
const TITLE_OBSERVATION_NAME = "generate-title";
const STEER_OBSERVATION_NAME = "steer";
const COMPACTION_OBSERVATION_NAME = "compact-context";
const BRANCH_SUMMARY_OBSERVATION_NAME = "summarize-branch";
const TOOL_USAGE_OBSERVATION_NAME = "tool-llm-usage";
const COMPACTION_RESULT_OBSERVATION_NAME = "compact-context-result";
const REQUEST_TOOL_APPROVAL_OBSERVATION_NAME = "request-tool-approval";
const RESOLVE_TOOL_APPROVAL_OBSERVATION_NAME = "resolve-tool-approval";
const RETRY_REQUEST_OBSERVATION_NAME = "retry-request";
const SETTLE_RETRY_OBSERVATION_NAME = "settle-retry";
const APPLY_MODEL_FALLBACK_OBSERVATION_NAME = "apply-model-fallback";
const CONFIRM_MODEL_FALLBACK_OBSERVATION_NAME = "confirm-model-fallback";
const TRIGGER_TTSR_OBSERVATION_NAME = "trigger-ttsr-rules";
const START_AUTO_COMPACTION_OBSERVATION_NAME = "start-auto-compaction";
const END_AUTO_COMPACTION_OBSERVATION_NAME = "end-auto-compaction";
const SWITCH_MODEL_OBSERVATION_NAME = "switch-model";
const SWITCH_ACCOUNT_OBSERVATION_NAME = "switch-account";
const EXPORT_EXCLUDED_ATTRIBUTE = "omp.langfuse.export_excluded";
const BASE_TAGS = ["omp"];
/**
 * How long an aborted run's turn stays open. OMP restarts the loop to deliver a
 * steer by aborting it, and its `agent_end` notification is fire-and-forget, so
 * the restart (or the steer's `before_agent_start`) can land just after it.
 */
export const RESTART_GRACE_MS = 2000;
/**
 * How long a turn whose `agent_end` promised an automatic continuation (retry
 * backoff, compaction, stop hook) waits for the restart before settling anyway.
 */
export const CONTINUATION_GRACE_MS = 5 * 60_000;

/** Read-only lookup tools are typed `retriever`; every other tool stays `tool`. */
const RETRIEVER_TOOLS: ReadonlySet<string> = new Set([
 "read", "grep", "glob", "find", "ast_grep", "web_search", "fetch", "lsp",
]);

/**
 * omp's title fork is a side turn whose last message is its `<title-request>`
 * prompt. Read off the wire payload (`messages`, Responses `input`, Gemini
 * `contents`) so the check holds on dialects the normalizer does not know.
 */
function isTitleRequest(payload: unknown): boolean {
 if (!payload || typeof payload !== "object") return false;
 const bag = payload as Record<string, unknown>;
 const list = [bag.messages, bag.input, bag.contents].find(Array.isArray);
 const last: unknown = list?.at(-1);
 if (!last || typeof last !== "object") return false;
 const body: unknown = "content" in last ? last.content : "parts" in last ? last.parts : undefined;
 if (typeof body === "string") return body.includes("<title-request>");
 // Anthropic/OpenAI tag parts `text`, Responses `input_text`, Gemini has no tag: read any `text` field.
 return Array.isArray(body) && body.some(part =>
  !!part && typeof part === "object" && "text" in part && typeof part.text === "string" && part.text.includes("<title-request>"),
 );
}

// A data: URI is only worth emitting if the processor swaps it for a media
// reference; with uploads off it stays in the span as full base64.
const EMIT_IMAGE_MEDIA = (() => {
 const raw = process.env.LANGFUSE_MEDIA_UPLOAD_ENABLED?.trim().toLowerCase();
 return raw ? !["false", "0"].includes(raw) : true;
})();

const ENV_TRACEPARENT = "LANGFUSE_PI_TRACEPARENT";
const ENV_PARENT_TRACE_ID = "LANGFUSE_PI_PARENT_TRACE_ID";
const ENV_PARENT_SPAN_ID = "LANGFUSE_PI_PARENT_SPAN_ID";
const ENV_PARENT_SESSION_ID = "LANGFUSE_PI_PARENT_SESSION_ID";
const ENV_PARENT_DEPTH = "LANGFUSE_PI_PARENT_DEPTH";
const ENV_PARENT_EXTERNAL_TRACE = "LANGFUSE_PI_PARENT_EXTERNAL_TRACE";

const HEX_TRACE_ID = /^[0-9a-f]{32}$/;
const HEX_SPAN_ID = /^[0-9a-f]{16}$/;

const isUsableTraceId = (v: string | undefined): v is string =>
 !!v && HEX_TRACE_ID.test(v) && !/^0+$/.test(v);
const isUsableSpanId = (v: string | undefined): v is string =>
 !!v && HEX_SPAN_ID.test(v) && !/^0+$/.test(v);

const TRACEPARENT = /^(?!ff)([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})(-.*)?$/;

export function parseTraceparent(value: string): { traceId: string; spanId: string } | undefined {
 const match = TRACEPARENT.exec(value.trim().toLowerCase());
 if (!match) return undefined;
 const [, version, traceId, spanId, , trailing] = match;
 if (version === "00" && trailing) return undefined;
 if (!isUsableTraceId(traceId) || !isUsableSpanId(spanId)) return undefined;
 return { traceId, spanId };
}

export interface InheritedParent {
 spanContext: SpanContext;
 sessionId?: string;
 depth: number;
 source: "subagent" | "attached";
 externalTrace: boolean;
}

export function readInheritedParent(env: NodeJS.ProcessEnv = process.env): InheritedParent | undefined {
 const parent = readParentIds(env);
 if (!parent) return undefined;
 const depth = Number(env[ENV_PARENT_DEPTH] ?? "0");
 return {
  spanContext: {
   traceId: parent.traceId,
   spanId: parent.spanId,
   traceFlags: TraceFlags.SAMPLED,
   isRemote: true,
  },
  sessionId: env[ENV_PARENT_SESSION_ID]?.trim() || undefined,
  depth: Number.isFinite(depth) && depth > 0 ? depth : 0,
  source: parent.source,
  externalTrace: parent.source === "attached" || env[ENV_PARENT_EXTERNAL_TRACE] === "1",
 };
}

type ParentIds = { traceId: string; spanId: string; source: "subagent" | "attached" };

function readParentIds(env: NodeJS.ProcessEnv): ParentIds | undefined {
 const raw = env[ENV_TRACEPARENT]?.trim();
 if (raw) {
  const parsed = parseTraceparent(raw);
  if (parsed) return { ...parsed, source: "attached" };
  console.error(`[pi-langfuse] Ignoring malformed ${ENV_TRACEPARENT}: ${JSON.stringify(raw)}`);
 }
 const traceId = env[ENV_PARENT_TRACE_ID]?.trim().toLowerCase();
 const spanId = env[ENV_PARENT_SPAN_ID]?.trim().toLowerCase();
 if (!traceId && !spanId) return undefined;
 if (!isUsableTraceId(traceId) || !isUsableSpanId(spanId)) return undefined;
 return { traceId, spanId, source: "subagent" };
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export interface LangfuseConfig {
 publicKey: string;
 secretKey: string;
 baseUrl: string;
 userId?: string;
 environment?: string;
 release?: string;
}
export function tracingEnabled(): boolean {
 return (process.env.LANGFUSE_TRACING_ENABLED ?? "").trim().toLowerCase() !== "false";
}


export function loadConfig(): LangfuseConfig | undefined {
 // Kill switch: wins over both env keys and the config file.
 if (!tracingEnabled()) {
  return undefined;
 }
 const file = readConfigFile();
 const asTrimmedString = (v: unknown): string | undefined =>
  typeof v === "string" && v.trim() ? v.trim() : undefined;

 const publicKey = process.env.LANGFUSE_PUBLIC_KEY?.trim() || asTrimmedString(file.publicKey);
 const secretKey = process.env.LANGFUSE_SECRET_KEY?.trim() || asTrimmedString(file.secretKey);
 if (!publicKey || !secretKey) return undefined;
 const baseUrl =
  process.env.LANGFUSE_BASE_URL?.trim() ||
  process.env.LANGFUSE_HOST?.trim() ||
  asTrimmedString(file.baseUrl) ||
  "https://cloud.langfuse.com";
 return {
  publicKey,
  secretKey,
  baseUrl: baseUrl.replace(/\/$/, ""),
  userId: process.env.LANGFUSE_USER_ID?.trim() || asTrimmedString(file.userId),
  environment: process.env.LANGFUSE_TRACING_ENVIRONMENT?.trim() || asTrimmedString(file.environment),
  release: process.env.LANGFUSE_RELEASE?.trim() || asTrimmedString(file.release),
 };
}

function readConfigFile(): Partial<Record<keyof LangfuseConfig, unknown>> {
 try {
  const path = join(getAgentDir(), "langfuse.json");
  if (!existsSync(path)) return {};
  const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
  return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : {};
 } catch (error) {
  console.error(
   `[pi-langfuse] Ignoring unreadable langfuse.json: ${(error as Error).message}`,
  );
  return {};
 }
}

// ---------------------------------------------------------------------------
// Payload helpers
// ---------------------------------------------------------------------------

const SECRET_REDACTION_MARK = "[redacted-langfuse-secret]";
const CYCLE_MARK = "[circular-ref]";
const LANGFUSE_KEY_TOKEN = String.raw`\b[sp]k-lf-[\w-]+\b`;

function escapeRegExpLiteral(text: string): string {
 return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function createSecretRedactor(...extraSecrets: string[]): (value: unknown) => unknown {
 return buildRedactor(extraSecrets);
}

function buildRedactor(extraSecrets: readonly string[], transform?: (text: string) => string): (value: unknown) => unknown {
 const alternatives = extraSecrets.filter((s) => s.length > 0).map(escapeRegExpLiteral);
 alternatives.push(LANGFUSE_KEY_TOKEN);
 const pattern = new RegExp(alternatives.join("|"), "g");
 const walk = (value: unknown, ancestors: readonly object[]): unknown => {
  if (typeof value === "string") {
   const redacted = value.replace(pattern, SECRET_REDACTION_MARK);
   return transform ? transform(redacted) : redacted;
  }
  if (value === null || typeof value !== "object") return value;
  // Only plain objects are rebuilt; a Date, Map, Set or class instance has no
  // enumerable own entries worth walking and would otherwise flatten to `{}`.
  const proto = Object.getPrototypeOf(value);
  if (!Array.isArray(value) && proto !== Object.prototype && proto !== null) return value;
  if (ancestors.includes(value)) return CYCLE_MARK;
  const chain = [...ancestors, value];
  if (Array.isArray(value)) {
   const items: unknown[] = [];
   for (const item of value) items.push(walk(item, chain));
   return items;
  }
  const fields: Record<string, unknown> = {};
  for (const [key, field] of Object.entries(value)) {
   Object.defineProperty(fields, key, {
    value: walk(field, chain),
    enumerable: true,
    writable: true,
    configurable: true,
   });
  }
  return fields;
 };
 return (value) => walk(value, []);
}

const redactLangfuseKeys = createSecretRedactor();
const redactToolDetails = buildRedactor([], markDataUris);

/** OMP returns the system prompt as its ordered sections; the provider receives them joined. */
export function readSystemPrompt(ctx: {
 getSystemPrompt?: () => string | readonly string[] | undefined;
}): string | undefined {
 let raw: unknown;
 try {
  raw = ctx.getSystemPrompt?.();
 } catch {
  return undefined;
 }
 const text = Array.isArray(raw) ? raw.filter((part): part is string => typeof part === "string").join("\n\n") : raw;
 if (typeof text !== "string" || !text.trim()) return undefined;
 return redactLangfuseKeys(text) as string;
}

export function extractText(content: unknown): string {
 if (typeof content === "string") return content;
 if (!Array.isArray(content)) return "";
 return content
  .filter((p): p is { type: string; text: string } => !!p && typeof p === "object" && (p as { type?: string }).type === "text")
  .map((p) => p.text)
  .join("");
}

/**
 * Langfuse renders an assistant tool call as a card only when `arguments` is a
 * JSON-encoded string, so the arguments travel with every call.
 */
export function extractToolCalls(content: unknown): ChatMlToolCall[] {
 if (!Array.isArray(content)) return [];
 return content
  .filter(
   (p): p is { type: "toolCall"; id: string; name: string; arguments?: unknown } =>
    !!p && typeof p === "object" && "type" in p && p.type === "toolCall",
  )
  .map(part => {
   const args = safeStringify(redactLangfuseKeys(part.arguments));
   return {
    id: part.id,
    type: "function" as const,
    function: { name: part.name, ...(args ? { arguments: markDataUris(args) } : {}) },
   };
  });
}

export interface ToolDefinitionInput {
 name: string;
 description?: string;
 parameters?: unknown;
}

export function activeToolDefinitions(
 pi: Pick<TracingAPI, "getAllTools" | "getActiveTools">,
): ToolDefinitionInput[] {
 try {
  const registry = new Map((pi.getAllTools?.() ?? []).map((tool) => [tool.name, tool]));
  const out: ToolDefinitionInput[] = [];
  const seen = new Set<string>();
  for (const name of pi.getActiveTools?.() ?? []) {
   const tool = registry.get(name);
   if (!tool || seen.has(name)) continue;
   seen.add(name);
   out.push({ name: tool.name, description: tool.description, parameters: tool.parameters });
  }
  return out;
 } catch {
  return [];
 }
}

export function attachToolDefinitions(input: unknown, tools: ToolDefinitionInput[]): unknown {
 if (!tools.length || !Array.isArray(input)) return input;
 const [first, ...rest] = input;
 return first && typeof first === "object" ? [{ ...first, tools }, ...rest] : input;
}

export interface PiImagePart {
 type: "image";
 data: string;
 mimeType: string;
}

export function extractImages(content: unknown): PiImagePart[] {
 if (!Array.isArray(content)) return [];
 return content.filter(
  (p): p is PiImagePart =>
   !!p &&
   typeof p === "object" &&
   (p as { type?: string }).type === "image" &&
   typeof (p as { data?: unknown }).data === "string" &&
   typeof (p as { mimeType?: unknown }).mimeType === "string",
 );
}

export function describeImage(image: { type?: unknown; data?: unknown; mimeType?: unknown }): string {
 const mime = typeof image.mimeType === "string" && image.mimeType ? image.mimeType : "unknown type";
 if (typeof image.data !== "string" || !image.data) return `[image ${mime}]`;
 const kb = Math.floor((image.data.length * 3) / 4 / 1024);
 return `[image ${mime} ~${kb}KB]`;
}

export function renderContentWithImageMarkers(content: unknown): string {
 if (typeof content === "string") return content;
 if (!Array.isArray(content)) return "";
 return content
  .map((p) => {
   if (!p || typeof p !== "object") return "";
   const part = p as { type?: string; text?: string };
   if (part.type === "text") return typeof part.text === "string" ? part.text : "";
   if (part.type === "image") return describeImage(part as { data?: unknown; mimeType?: unknown });
   return "";
  })
  .filter(Boolean)
  .join("\n");
}

export type ContentPart =
 | { type: "text"; text: string }
 | { type: "image_url"; image_url: { url: string } };

/**
 * pi hands us raw base64. The processor's media matcher accepts a valid prefix
 * of malformed base64, which would upload a corrupt file; validate it first.
 */
export function toDataUri(image: PiImagePart): string | undefined {
 const data = image.data.replace(/\s+/g, "");
 if (!data || !/^[A-Za-z0-9+/]+={0,2}$/.test(data)) return undefined;
 if (!image.mimeType || image.mimeType.includes(";")) return undefined;
 return `data:${image.mimeType};base64,${data}`;
}

export function toMultimodalContent(
 text: string,
 images: readonly PiImagePart[] | undefined,
): string | ContentPart[] {
 const urls = (images ?? []).map(toDataUri).filter((url): url is string => !!url);
 if (!urls.length) return text;
 return [
  ...(text ? [{ type: "text" as const, text }] : []),
  ...urls.map((url) => ({ type: "image_url" as const, image_url: { url } })),
 ];
}

const MAX_TOKENS_KEYS = ["max_tokens", "max_completion_tokens", "max_output_tokens", "maxOutputTokens", "maxTokens"];
const MAX_PARAM_CHARS = 200;
const THINKING_BUDGET_KEYS = [
 "thinking_token_budget",
 "thinking_budget",
 "thinking_budget_tokens",
 "budget_tokens",
 "thinkingBudget",
];
/**
 * Sampling settings as the provider dialects spell them on the wire. OMP models
 * do not declare their sampling parameters, so the payload is the source.
 */
const SAMPLING_KEYS = [
 "temperature",
 "top_p",
 "topP",
 "top_k",
 "topK",
 "min_p",
 "presence_penalty",
 "frequency_penalty",
 "repetition_penalty",
 "seed",
 "stop",
 "stop_sequences",
 "reasoning_effort",
 "verbosity",
];

/** Containers the dialects nest caps and budgets in; nothing else is entered, so arbitrary payload values (and their getters) stay unread. */
const PARAM_CONTAINER_KEYS = [
 "config", "generationConfig", "thinkingConfig", "inferenceConfig", "additionalModelRequestFields", "thinking", "options",
];

function findNumber(bag: unknown, keys: string[], depth = 2): number | undefined {
 if (!bag || typeof bag !== "object" || Array.isArray(bag)) return undefined;
 const record = bag as Record<string, unknown>;
 for (const key of keys) {
  const value = record[key];
  if (typeof value === "number" && Number.isInteger(value) && value > 0) return value;
 }
 if (depth === 0) return undefined;
 for (const key of PARAM_CONTAINER_KEYS) {
  if (!(key in record)) continue;
  const found = findNumber(record[key], keys, depth - 1);
  if (found !== undefined) return found;
 }
 return undefined;
}

function pickCacheRetention(payload: unknown): string | undefined {
 const bag = payload as Record<string, unknown> | undefined;
 const flat = bag?.prompt_cache_retention;
 if (typeof flat === "string") return flat;
 const system = bag?.system;
 if (!Array.isArray(system)) return undefined;
 for (const block of system) {
  const ttl = (block as { cache_control?: { ttl?: unknown } } | undefined)?.cache_control?.ttl;
  if (typeof ttl === "string") return ttl;
 }
 return undefined;
}

function pickToolChoice(payload: unknown): string | undefined {
 const value = (payload as Record<string, unknown> | undefined)?.tool_choice;
 if (typeof value === "string") return value;
 const type = (value as { type?: unknown } | undefined)?.type;
 return typeof type === "string" ? type : undefined;
}

export function extractModelParameters(
 payload: unknown,
 model: { reasoning?: boolean; samplingParams?: Record<string, unknown> } | undefined,
 thinkingLevel?: string,
): Record<string, string | number> | undefined {
 const out: Record<string, string | number> = {};
 try {
  const maxTokens = findNumber(payload, MAX_TOKENS_KEYS);
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (model?.reasoning && thinkingLevel && thinkingLevel !== "off") out.thinking_level = thinkingLevel;
  const thinkingBudget = findNumber(payload, THINKING_BUDGET_KEYS);
  if (thinkingBudget !== undefined) out.thinking_budget_tokens = thinkingBudget;
  const cacheRetention = pickCacheRetention(payload);
  if (cacheRetention !== undefined) out.prompt_cache_retention = cacheRetention;
  const serviceTier = (payload as Record<string, unknown> | undefined)?.service_tier;
  if (typeof serviceTier === "string") out.service_tier = serviceTier;
  const toolChoice = pickToolChoice(payload);
  if (toolChoice !== undefined) out.tool_choice = toolChoice;
  for (const key of new Set([...SAMPLING_KEYS, ...Object.keys(model?.samplingParams ?? {})])) {
   try {
    const value = (payload as Record<string, unknown> | undefined)?.[key];
    if (value === undefined) continue;
    if (typeof value === "number") out[key] = value;
    else {
     let remaining = MAX_PARAM_CHARS;
     const text = typeof value === "string" ? value : JSON.stringify(value, (field, part: unknown) => {
      remaining -= field.length + (typeof part === "string" ? part.length : 1);
      if (remaining < 0) throw new Error("Model parameter exceeds the serialization budget");
      return part;
     });
     if (text !== undefined && text.length <= MAX_PARAM_CHARS) out[key] = text;
    }
   } catch {
    debug("model parameter omitted", key);
   }
  }
 } catch { debug("model parameter extraction unavailable"); }
 return Object.keys(out).length ? out : undefined;
}

type PiAgentMessages = Parameters<typeof convertToLlm>[0];

export type TracingHistoryPart =
 | { type: "text"; text: string }
 | PiImagePart
 | { type: "thinking"; thinking: string }
 | { type: "redactedThinking"; data: string }
 | { type: "toolCall"; id: string; name: string; arguments: unknown };

/** Provider-visible history; replay must not invent usage, model, or timestamps for old messages. */
export interface TracingHistoryMessage {
 role: "user" | "assistant" | "toolResult";
 content: string | TracingHistoryPart[];
 toolCallId?: string;
 toolName?: string;
 isError?: boolean;
}

/** Assistant facts consumed by tracing; archives need not invent the SDK's other fields. */
export interface TracingAssistantMessage {
 role: "assistant";
 content: TracingHistoryPart[];
 api: string;
 provider: string;
 model: string;
 usage?: PiUsage;
 stopReason: "stop" | "length" | "toolUse" | "error" | "aborted";
 upstreamModel?: string;
 upstreamProvider?: string;
 responseId?: string;
 errorMessage?: string;
 errorStatus?: number;
 ttft?: number;
 duration?: number;
 /** When omp sent the request this message answers; pairs the message with its generation. */
 timestamp?: number;
 disabledFeatures?: string[];
 /** Stored credential row (account) that served the reply. */
 credentialId?: number;
}

export type TracingHistory = PiAgentMessages | TracingHistoryMessage[];
export type TracingEvent =
 | Exclude<ExtensionEvent, { type: "context" | "message_end" }>
 | (Omit<Extract<ExtensionEvent, { type: "context" }>, "messages"> & { messages: TracingHistory })
 | { type: "message_end"; message: Extract<ExtensionEvent, { type: "message_end" }>["message"] | TracingAssistantMessage };

export interface ChatMlToolCall {
 id: string;
 type: "function";
 function: { name: string; arguments?: string };
}

export interface ChatMlThinkingPart {
 type: "thinking";
 content: string;
 redacted?: true;
}

export type ChatMlMessage =
 | { role: "user"; content: string }
 | {
  role: "assistant";
  content?: string;
  thinking?: ChatMlThinkingPart[];
  tool_calls?: ChatMlToolCall[];
 }
 | { role: "tool"; tool_call_id: string; name?: string; content: string; is_error?: true };

export function markDataUris(text: string): string {
 return text.replace(
  /data:[^;,]{0,100};base64,[A-Za-z0-9+/]+=*/g,
  (uri) => `[data uri ~${Math.floor((uri.length * 3) / 4 / 1024)}KB]`,
 );
}

function renderHistoryContent(content: unknown): string {
 return typeof content === "string" || Array.isArray(content)
  ? renderContentWithImageMarkers(content)
  : safeStringify(content);
}

/**
 * Every thinking block, exactly as the provider returned it: summarized or full
 * `thinking` text, and the opaque `data` of a `redactedThinking` block.
 */
export function extractThinking(content: unknown): ChatMlThinkingPart[] {
 if (!Array.isArray(content)) return [];
 const parts: ChatMlThinkingPart[] = [];
 for (const part of content) {
  if (!part || typeof part !== "object" || !("type" in part)) continue;
  if (part.type === "thinking" && "thinking" in part && typeof part.thinking === "string") {
   parts.push({ type: "thinking", content: markDataUris(part.thinking) });
  } else if (part.type === "redactedThinking" && "data" in part && typeof part.data === "string") {
   parts.push({ type: "thinking", content: part.data, redacted: true });
  }
 }
 return parts;
}

export function toChatMlMessage(message: unknown): ChatMlMessage | undefined {
 if (!message || typeof message !== "object") return undefined;
 const msg = message as {
  role?: unknown;
  content?: unknown;
  toolCallId?: unknown;
  toolName?: unknown;
  isError?: unknown;
 };
 if (msg.role === "user") {
  return { role: "user", content: markDataUris(renderHistoryContent(msg.content)) };
 }
 if (msg.role === "assistant") {
  const content = markDataUris(extractText(msg.content));
  const thinking = extractThinking(msg.content);
  const toolCalls = extractToolCalls(msg.content);
  if (!content && !thinking.length && !toolCalls.length) return undefined;
  return {
   role: "assistant",
   ...(content ? { content } : {}),
   ...(thinking.length ? { thinking } : {}),
   ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
  };
 }
 if (msg.role === "toolResult") {
  return {
   role: "tool",
   tool_call_id: typeof msg.toolCallId === "string" ? msg.toolCallId : "",
   ...(typeof msg.toolName === "string" && msg.toolName.length > 0 ? { name: msg.toolName } : {}),
   content: markDataUris(renderHistoryContent(msg.content)),
   ...(msg.isError ? { is_error: true as const } : {}),
  };
 }
 return undefined;
}

export function buildHistoryInput(messages: TracingHistory): ChatMlMessage[] | undefined {
 try {
  // OMP requires a string for its tool-name canonicalizer. Keep the compatibility
  // value local to conversion; toChatMlMessage omits it instead of exporting a
  // blank name. The archived/provider-visible messages remain unchanged.
  const sdkMessages = messages.map(message =>
   message.role === "toolResult" && !message.toolName ? { ...message, toolName: "" } : message,
  ) as PiAgentMessages;
  const history = convertToLlm(sdkMessages)
   .map((message) => toChatMlMessage(message))
   .filter((message): message is ChatMlMessage => message !== undefined);
  return history.length ? history : undefined;
 } catch {
  return undefined;
 }
}

/** The subset of OMP's `Usage` (`@oh-my-pi/pi-catalog`) the trace reports. */
export interface PiUsage {
 input: number;
 output: number;
 cacheRead: number;
 cacheWrite: number;
 /** Subset of `output`; undefined means the provider did not report it, not zero. */
 reasoningTokens?: number;
 /** Anthropic cache-write TTL split; the components sum to `cacheWrite`. */
 cttl?: { ephemeral5m?: number; ephemeral1h?: number };
 /** Server-side tool requests (web search/fetch) billed on this call. */
 server?: { webSearch?: number; webFetch?: number };
 cost?: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number };
}

/**
 * Whether the reasoning tokens can be split out of `output`.
 *
 * OMP reports `reasoningTokens` as a subset of `output`. A provider that reports
 * more reasoning than output would make the split produce a negative bucket, so
 * both {@link buildUsageDetails} and {@link buildCostDetails} fall back to a
 * single `output` bucket in that case. They MUST agree, otherwise a usage bucket
 * ends up without its cost counterpart.
 */
function resolveReasoningSplit(usage: PiUsage): { reasoning: number; canSplit: boolean } {
 const reasoning = usage.reasoningTokens ?? 0;
 return { reasoning, canSplit: reasoning > 0 && reasoning <= usage.output };
}

/** Usage facts Langfuse has no usage key for; they ride along as metadata. */
function usageMetadata(usage: PiUsage | undefined): Record<string, number> {
 if (!usage) return {};
 return {
  ...(usage.cttl?.ephemeral5m ? { cache_write_5m_tokens: usage.cttl.ephemeral5m } : {}),
  ...(usage.cttl?.ephemeral1h ? { cache_write_1h_tokens: usage.cttl.ephemeral1h } : {}),
  ...(usage.server?.webSearch ? { server_web_search_requests: usage.server.webSearch } : {}),
  ...(usage.server?.webFetch ? { server_web_fetch_requests: usage.server.webFetch } : {}),
 };
}

/**
 * Changes the pi usage counts into Langfuse usage details. Langfuse counts each
 * token in one key only, but pi includes the reasoning tokens in `output`.
 */
export function buildUsageDetails(usage: PiUsage): Record<string, number> | undefined {
 const details: Record<string, number> = {};
 if (usage.input > 0) details.input = usage.input;
 const { reasoning, canSplit } = resolveReasoningSplit(usage);
 const output = canSplit ? usage.output - reasoning : usage.output;
 if (output > 0) details.output = output;
 if (canSplit) details.output_reasoning_tokens = reasoning;
 if (usage.cacheRead > 0) details.cache_read_input_tokens = usage.cacheRead;
 if (usage.cacheWrite > 0) details.cache_creation_input_tokens = usage.cacheWrite;
 return Object.keys(details).length ? details : undefined;
}

/**
 * Cost keys must mirror the usage keys ({@link buildUsageDetails}) — Langfuse
 * joins the two by name, and server-side pricing emits the same spellings. A
 * usage bucket without its cost twin makes the bucket's implied per-token rate
 * wrong in the UI even though the total stays correct.
 */
export function buildCostDetails(usage: PiUsage): Record<string, number> | undefined {
 const cost = usage.cost;
 if (!cost || !(cost.total > 0)) return undefined;
 const details: Record<string, number> = { total: cost.total };
 if (cost.input > 0) details.input = cost.input;
 if (cost.output > 0) {
  const { reasoning, canSplit } = resolveReasoningSplit(usage);
  if (canSplit) {
   const reasoningCost = cost.output * (reasoning / usage.output);
   const nonReasoningCost = cost.output - reasoningCost;
   if (nonReasoningCost > 0) details.output = nonReasoningCost;
   if (reasoningCost > 0) details.output_reasoning_tokens = reasoningCost;
  } else {
   details.output = cost.output;
  }
 }
 if (cost.cacheRead > 0) details.cache_read_input_tokens = cost.cacheRead;
 if (cost.cacheWrite > 0) details.cache_creation_input_tokens = cost.cacheWrite;
 return details;
}

// ---------------------------------------------------------------------------
// Runtime: one isolated OTel provider per process, never the global provider
// ---------------------------------------------------------------------------

export type TracingContext = Pick<ExtensionContext, "cwd" | "hasUI" | "getSystemPrompt"> & {
 agent: Pick<ExtensionAgentIdentity, "id" | "kind" | "parentId"> &
 Partial<Pick<ExtensionAgentIdentity, "name" | "depth">>;
 model?: Pick<NonNullable<ExtensionContext["model"]>, "id" | "provider" | "api"> &
 Partial<Pick<NonNullable<ExtensionContext["model"]>, "contextWindow" | "reasoning">> & {
  samplingParams?: Record<string, unknown>;
 };
 sessionManager: {
  getSessionId(): string;
  getEntries(): Array<{ type?: string; message?: { role?: string; content?: unknown } }>;
  /** Active-branch entries (root to leaf); absent on hosts and archives without a session tree. */
  getBranch?(): Array<{ type?: string; message?: { role?: string; content?: unknown } }>;
  getSessionName?(): string | undefined;
 };
 ui?: Pick<ExtensionContext["ui"], "setStatus">;
};

export interface TracingAPI {
 on<E extends TracingEvent["type"]>(
  type: E,
  handler: (event: Extract<TracingEvent, { type: E }>, ctx: TracingContext) => void | Promise<void>,
 ): void;
 getThinkingLevel(): string | undefined;
 getAllTools(): ToolDefinitionInput[];
 getActiveTools(): string[];
 exec?: ExtensionAPI["exec"];
}

export interface TracingOptions {
 now?: () => Date;
 /** Live SDK access; never supplied by archive replay. */
 sdk?: ExtensionAPI;
 /** Observe actual SDK identities, including inherited trace IDs. */
 onObservation?: (name: string, context: SpanContext) => void;
 replay?: {
  namespace: string;
  eventId(): string;
  metadata(): Record<string, unknown>;
  /** Recovery scope; intersects the normal Langfuse span/export gates. */
  includeObservationIds?: ReadonlySet<string>;
 };
}

type TraceAttributes = Record<string, string | string[]>;

interface MaintenanceAttributes {
 output?: { role: string; content: string };
 model?: string;
 usageDetails?: Record<string, number>;
 costDetails?: Record<string, number>;
 metadata: Record<string, unknown>;
}
interface MaintenanceOptions {
 startedAt?: Date;
 endedAt?: Date;
 parentAtStart?: SpanContext | null;
 asType?: "generation" | "event";
}

interface Runtime {
 processor: LangfuseSpanProcessor;
 provider: NodeTracerProvider;
 shutdown: boolean;
 config: LangfuseConfig;
}

interface OpenGeneration {
 obs: LangfuseGeneration;
 index: number;
 startedAt: Date;
 sawFirstToken: boolean;
 /** An assistant message is streaming for this request; a late `turn_end` must not sweep it. */
 bound: boolean;
 finished: boolean;
}

interface OpenTool {
 obs: LangfuseTool | LangfuseRetriever;
 name: string;
 startedAt: Date;
 /** Subagent names this call dispatches (`task` `tasks[].name`). */
 spawnNames: string[];
}

interface PromptState {
 agent: TracingContext["agent"];
 sessionId: string;
 registryKey: string;
 root: LangfuseAgent;
 turnNumber: number;
 generationCount: number;
 /** Requests sent but not yet answered, oldest first. Side requests omp sends while a reply streams stay here until the turn ends. */
 openGenerations: OpenGeneration[];
 openTools: Map<string, OpenTool>;
 pendingToolResults: Array<{ tool_call_id: string; name: string; content: string }>;
 lastAssistantText?: string;
 sawError: boolean;
 userText: string;
 turnImages: PiImagePart[];
 systemPrompt?: string;
 /** Runs of this prompt that reported `agent_start` but not yet `agent_end`. */
 runsOpen: number;
 /** The last run ended but OMP is expected to start another for this prompt (retry, steer restart). */
 awaitingRestart: boolean;
 /** A steer arrived and no loop has started or progressed since: its restart may still be coming. */
 steerPendingStart: boolean;
 /** Finalizes an aborted turn if no restart arrives within {@link RESTART_GRACE_MS}. */
 settleTimer?: NodeJS.Timeout;
 /** The user interrupted at least one run. An interrupt is not an error. */
 interrupted: boolean;
 steers: number;
 promptSequence: number;
}

const DEBUG = process.env.PI_LANGFUSE_DEBUG === "true";
const debug = (...args: unknown[]) => {
 if (DEBUG) console.error("[pi-langfuse]", ...args);
};

// OMP rebinds this factory into every in-process subagent session (task, eval
// `agent()`, `/tan` clones), and module-level state is shared across them. The
// exporter, the per-trace attributes and the live turns therefore live here, so
// a subagent's spans land in its parent's trace through the same exporter.
let runtime: Runtime | undefined;
const activeTracingSessions = new Set<object>();
const runtimeRetirements: Promise<void>[] = [];
const traceAttributesByTraceId = new Map<string, TraceAttributes>();
let pendingRootAttributes: TraceAttributes | undefined;
const MAX_TRACKED_TRACES = 256;
let pendingObservationSeed: string | undefined;
const randomIds = new RandomIdGenerator();
const tracingIds: IdGenerator = {
 generateTraceId: () => pendingObservationSeed
  ? digest(`omp-backfill-v1:${pendingObservationSeed}:trace`).slice(0, 32)
  : randomIds.generateTraceId(),
 generateSpanId: () => pendingObservationSeed
  ? digest(`omp-backfill-v1:${pendingObservationSeed}:span`).slice(0, 16)
  : randomIds.generateSpanId(),
};
/** Live turn per OMP agent registry id; a subagent finds its parent's turn here. */
const liveTurns = new Map<string, PromptState>();
let gitBranch: string | undefined;
let sessionHadImages = false;

// A parent in another process (an `omp` launched from a tool, or an external
// trace via LANGFUSE_PI_TRACEPARENT) arrives through the environment. Snapshot
// it at load: this process publishes its own values into the same variables.
const INHERITED_PARENT_ENV: Record<string, string | undefined> = {
 [ENV_TRACEPARENT]: process.env[ENV_TRACEPARENT],
 [ENV_PARENT_TRACE_ID]: process.env[ENV_PARENT_TRACE_ID],
 [ENV_PARENT_SPAN_ID]: process.env[ENV_PARENT_SPAN_ID],
 [ENV_PARENT_SESSION_ID]: process.env[ENV_PARENT_SESSION_ID],
 [ENV_PARENT_DEPTH]: process.env[ENV_PARENT_DEPTH],
 [ENV_PARENT_EXTERNAL_TRACE]: process.env[ENV_PARENT_EXTERNAL_TRACE],
};
const INHERITED_PARENT = readInheritedParent(INHERITED_PARENT_ENV);

function isLangfuseSpan(span: { attributes: Record<string, unknown> }): boolean {
 return typeof span.attributes["langfuse.observation.type"] === "string";
}

function createRuntime(config: LangfuseConfig): Runtime {
 const redactSecrets = createSecretRedactor(config.publicKey, config.secretKey);
 const processor = new LangfuseSpanProcessor({
  publicKey: config.publicKey,
  secretKey: config.secretKey,
  baseUrl: config.baseUrl,
  environment: config.environment,
  release: config.release,
  mask: ({ data }) => redactSecrets(data),
  shouldExportSpan: ({ otelSpan }) => isLangfuseSpan(otelSpan) &&
   otelSpan.attributes[EXPORT_EXCLUDED_ATTRIBUTE] !== true,
 });
 // Langfuse reads trace fields per span, and an extension must not install the
 // global OTel context manager that propagateAttributes() needs. Each span is
 // stamped with the fields of the trace it belongs to, so concurrent turns and
 // subagents sharing this processor never borrow each other's session.
 const baseOnStart = processor.onStart.bind(processor);
 processor.onStart = (span, parentContext) => {
  baseOnStart(span, parentContext);
  const traceId = span.spanContext().traceId;
  let attributes = traceAttributesByTraceId.get(traceId);
  if (!attributes && pendingRootAttributes) {
   attributes = pendingRootAttributes;
   if (traceAttributesByTraceId.size >= MAX_TRACKED_TRACES) {
    const oldest = traceAttributesByTraceId.keys().next().value;
    if (oldest !== undefined) traceAttributesByTraceId.delete(oldest);
   }
   traceAttributesByTraceId.set(traceId, attributes);
  }
  if (attributes) span.setAttributes(attributes);
 };
 const provider = new NodeTracerProvider({
  // OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES still win over the fallback name.
  resource: defaultResource()
   .merge(resourceFromAttributes({ "service.name": "omp" }))
   .merge(detectResources({ detectors: [envDetector] })),
  spanProcessors: [processor],
  idGenerator: tracingIds,
  sampler: new AlwaysOnSampler(),
  spanLimits: { attributeValueLengthLimit: Infinity, attributeCountLimit: Infinity },
 });
 setLangfuseTracerProvider(provider);
 return { processor, provider, shutdown: false, config: { ...config } };
}

function ensureRuntime(config: LangfuseConfig): Runtime {
 if (runtime && !runtime.shutdown) {
  const previous = runtime.config;
  if (previous.publicKey === config.publicKey && previous.secretKey === config.secretKey &&
   previous.baseUrl === config.baseUrl && previous.environment === config.environment && previous.release === config.release) return runtime;
  if (activeTracingSessions.size > 0 || liveTurns.size > 0) {
   throw new Error("Langfuse runtime configuration conflicts with an active tracing session");
  }
  const retiring = runtime;
  retiring.shutdown = true;
  const drain = retiring.processor.forceFlush().then(() => retiring.provider.shutdown());
  // Keep the original rejection for strict offline drains, but do not leave an
  // unhandled rejection while the next provider is being used.
  void drain.catch(() => { });
  runtimeRetirements.push(drain);
 }
 runtime = createRuntime(config);
 return runtime;
}

/** Starts a trace root; a new trace adopts `attributes`, an existing trace keeps its own. */
function startTraceRoot<T>(attributes: TraceAttributes, start: () => T): T {
 pendingRootAttributes = attributes;
 try {
  return start();
 } finally {
  pendingRootAttributes = undefined;
 }
}

function traceAttributesFor(
 config: LangfuseConfig,
 opts: { name?: string; sessionId: string; external: boolean; backfill?: boolean; title?: string },
): TraceAttributes {
 if (opts.external) return {};
 const tags = opts.backfill ? [...BASE_TAGS, "backfill"] : [...BASE_TAGS];
 // Langfuse has no session-name field; the title tag makes OMP's session title searchable.
 if (opts.title) tags.push(`title:${opts.title.slice(0, 120)}`);
 return {
  ...(opts.name ? { [LangfuseOtelSpanAttributes.TRACE_NAME]: opts.name } : {}),
  [LangfuseOtelSpanAttributes.TRACE_SESSION_ID]: opts.sessionId,
  [LangfuseOtelSpanAttributes.TRACE_TAGS]: tags,
  ...(config.userId ? { [LangfuseOtelSpanAttributes.TRACE_USER_ID]: config.userId } : {}),
 };
}

// Flushing must never block OMP's exit or the next prompt indefinitely. The
// SDK's forceFlush awaits pending media uploads before the span export, so a
// session that uploaded images gets a larger budget on the exit path — exiting
// would abort an in-flight upload and leave a media token without its binary.
// Mid-session, timing out is harmless: the upload finishes in the background.
const FLUSH_TIMEOUT_MS = 3000;
const EXIT_FLUSH_WITH_MEDIA_TIMEOUT_MS = 15000;

async function withTimeout(work: Promise<unknown>, budgetMs: number): Promise<"done" | "timeout"> {
 let timer: NodeJS.Timeout | undefined;
 try {
  return await Promise.race([
   work.then(() => "done" as const),
   new Promise<"timeout">(resolve => {
    timer = setTimeout(() => resolve("timeout"), budgetMs);
    timer.unref?.();
   }),
  ]);
 } finally {
  clearTimeout(timer);
 }
}

async function flush(budgetMs: number = FLUSH_TIMEOUT_MS): Promise<void> {
 if (!runtime) return;
 try {
  const outcome = await withTimeout(flushTracing(), budgetMs);
  if (outcome === "timeout" && sessionHadImages) {
   console.error(
    `[pi-langfuse] flush timed out after ${budgetMs}ms with media pending — images may be missing from the trace`,
   );
  }
 } catch {
  // Tracing must never break the session.
 }
}

/** Strict drain for offline replay: unlike the host's best-effort flush, errors reach the caller. */
export async function flushTracing(): Promise<void> {
 await Promise.all(runtimeRetirements.splice(0));
 await runtime?.processor.forceFlush();
}

export async function shutdownTracing(): Promise<void> {
 await flushTracing();
 if (!runtime || runtime.shutdown || activeTracingSessions.size > 0 || liveTurns.size > 0) return;
 const current = runtime;
 current.shutdown = true;
 await current.provider.shutdown();
}


// Parent context for an `omp` process launched from this one. In-process
// subagents resolve their parent through `liveTurns`, never through these.
function publishParentContext(root: LangfuseAgent, sessionId: string): void {
 const spanContext = root.otelSpan.spanContext();
 // A root the sampler dropped has no exported span for children to point at.
 if (!(spanContext.traceFlags & TraceFlags.SAMPLED)) return;
 delete process.env[ENV_TRACEPARENT];
 process.env[ENV_PARENT_TRACE_ID] = spanContext.traceId;
 process.env[ENV_PARENT_SPAN_ID] = spanContext.spanId;
 process.env[ENV_PARENT_SESSION_ID] = sessionId;
 process.env[ENV_PARENT_DEPTH] = String((INHERITED_PARENT?.depth ?? 0) + 1);
 if (INHERITED_PARENT?.externalTrace) process.env[ENV_PARENT_EXTERNAL_TRACE] = "1";
 else delete process.env[ENV_PARENT_EXTERNAL_TRACE];
}

// A child must not attach to a turn that has ended. The inherited values stay
// valid, so put them back instead of a plain delete. Another live main session
// in this process may have published its own turn since, which stays.
function withdrawParentContext(root: LangfuseAgent): void {
 if (process.env[ENV_PARENT_SPAN_ID] !== root.otelSpan.spanContext().spanId) return;
 for (const [key, value] of Object.entries(INHERITED_PARENT_ENV)) {
  if (value === undefined) delete process.env[key];
  else process.env[key] = value;
 }
}

/** Names of the subagents a tool call dispatches: OMP's `task` tool takes `tasks[].name`. */
function spawnedAgentNames(args: unknown): string[] {
 if (!args || typeof args !== "object" || !("tasks" in args) || !Array.isArray(args.tasks)) return [];
 const names: string[] = [];
 for (const task of args.tasks) {
  if (task && typeof task === "object" && "name" in task && typeof task.name === "string") names.push(task.name);
 }
 return names;
}

/**
 * The span an in-process subagent's turn nests under: the parent's tool call
 * that dispatched it, or the parent's turn when the dispatching call cannot be
 * told apart from other open calls.
 */
function resolveSubagentParent(
 agent: TracingContext["agent"],
 scope = "",
): { parent: PromptState; spanContext: SpanContext } | undefined {
 if (agent.kind !== "sub" || !agent.parentId) return undefined;
 const parent = liveTurns.get(`${scope}${agent.parentId}`);
 if (!parent) return undefined;
 const tools = [...parent.openTools.values()];
 const dispatch =
  tools.find(tool =>
   tool.spawnNames.some(
    name => agent.id === name || agent.id.endsWith(`-${name}`) || agent.id.endsWith(`.${name}`),
   ),
  ) ?? (tools.length === 1 ? tools[0] : undefined);
 return { parent, spanContext: (dispatch?.obs ?? parent.root).otelSpan.spanContext() };
}

function isUsage(value: unknown): value is PiUsage {
 return (
  !!value &&
  typeof value === "object" &&
  "input" in value &&
  typeof value.input === "number" &&
  "output" in value &&
  typeof value.output === "number"
 );
}

export default function langfuseObservability(pi: ExtensionAPI) {
 debug("factory start");
 // Accounting registers first so raw capture and tracing read its per-reply results.
 registerCacheAccounting(pi);
 // Local capture does not depend on Langfuse, so it registers before the no-keys exit.
 registerRawCapture(pi);
 const config = loadConfig();
 if (!config) {
  pi.on("session_start", (_event, ctx) => {
   if (ctx.hasUI) ctx.ui.setStatus("langfuse", "langfuse: off (no keys)");
  });
  return;
 }

 registerTracing(pi, config, { sdk: pi });
}

/** The single tracing consumer used by live extensions and historical replay. */
export function registerTracing(pi: TracingAPI, config: LangfuseConfig, options: TracingOptions = {}): void {
 if (!tracingEnabled()) return;
 // Explicit config cannot bypass the consumer's tracing kill switch.
 const sessionToken = {};
 const selectedIds = options.replay?.includeObservationIds
  ? new Set(options.replay.includeObservationIds) : undefined;
 const now = options.now ?? (() => new Date(Date.now()));
 const registryScope = options.replay ? `${options.replay.namespace}:` : "";
 const observe = <T extends LangfuseObservation>(key: string, create: () => T): T => {
  if (!options.replay) {
   const observation = create();
   options.onObservation?.(key, observation.otelSpan.spanContext());
   return observation;
  }
  const previous = pendingObservationSeed;
  pendingObservationSeed = `${registryScope}${options.replay.eventId()}:${key}`;
  try {
   const observation = create();
   if (selectedIds && !selectedIds.has(observation.otelSpan.spanContext().spanId)) {
    observation.otelSpan.setAttribute(EXPORT_EXCLUDED_ATTRIBUTE, true);
   }
   options.onObservation?.(key, observation.otelSpan.spanContext());
   return observation;
  } finally {
   pendingObservationSeed = previous;
  }
 };
 const observationAttributes = <T extends LangfuseObservationAttributes>(value: T): T =>
  options.replay
   ? { ...value, metadata: { ...value.metadata, backfill: true, ...options.replay.metadata() } }
   : value;


 // Per session: OMP binds a fresh API, and so a fresh closure, for every session.
 let state: PromptState | undefined;
 let fallbackTurnCounter = 0;
 let promptSequence = 0;
 let messageOwners = new WeakMap<object, number>();
 // Runs a newer prompt superseded before their agent_end arrived, and runs that
 // started with no turn open (their first provider request opens a continuation).
 let supersededRunsOpen = 0;
 let unclaimedRunStarts = 0;
 let lastPromptText = "";
 let lastContextHistory: ChatMlMessage[] | undefined;
 let compactionStartedAt: Date | undefined;
 let compactionReason: string | undefined;
 let sdkCompactionCalls = 0;

 // Only the active branch counts: after navigating back and prompting again,
 // the abandoned branch's prompts are still stored but no longer in the conversation.
 const activeEntries = (ctx: TracingContext) =>
  typeof ctx.sessionManager.getBranch === "function" ? ctx.sessionManager.getBranch() : ctx.sessionManager.getEntries();

 const resolveTurnNumber = (ctx: TracingContext, promptText: string): number => {
  try {
   const entries = activeEntries(ctx);
   const userMessages = entries.filter(e => e.type === "message" && e.message?.role === "user");
   let count = userMessages.length;
   const last = userMessages[userMessages.length - 1];
   // before_agent_start may fire before or after the prompt is appended.
   if (last && extractText(last.message?.content) !== promptText) count += 1;
   if (count < 1) count = 1;
   fallbackTurnCounter = count;
   return count;
  } catch {
   fallbackTurnCounter += 1;
   return fallbackTurnCounter;
  }
 };

 const closeDanglingObservations = (reason: "interrupted" | "superseded") => {
  if (!state) return;
  for (const tool of state.openTools.values()) {
   tool.obs.update({ level: "WARNING", statusMessage: `Tool run ${reason}`, metadata: { [reason]: true } });
   tool.obs.end(now());
  }
  state.openTools.clear();
  for (const gen of state.openGenerations) {
   if (gen.finished) continue;
   gen.obs.update({ level: "WARNING", statusMessage: `Generation ${reason}`, metadata: { [reason]: true } });
   gen.obs.end(now());
   gen.finished = true;
  }
  state.openGenerations = [];
 };

 const finalizeRoot = (opts: { cancelled: boolean; continuationMissing?: boolean }) => {
  if (!state) return;
  clearTimeout(state.settleTimer);
  closeDanglingObservations("interrupted");
  const media = EMIT_IMAGE_MEDIA ? state.turnImages : [];
  if (media.length) sessionHadImages = true;
  state.root.update({
   input: media.length ? { role: "user", content: toMultimodalContent(state.userText, media) } : undefined,
   output: state.lastAssistantText ? { role: "assistant", content: state.lastAssistantText } : undefined,
   level: state.sawError ? "ERROR" : undefined,
   metadata: {
    ...(state.turnImages.length ? { image_count: state.turnImages.length } : {}),
    ...(state.steers ? { steer_count: state.steers } : {}),
    ...(state.interrupted ? { interrupted: true } : {}),
    ...(opts.cancelled ? { cancelled: true } : {}),
    ...(opts.continuationMissing ? { continuation_missing: true } : {}),
   },
  });
  state.root.end(now());
  if (liveTurns.get(state.registryKey) === state) liveTurns.delete(state.registryKey);
  if (state.agent.kind === "main" && !options.replay) withdrawParentContext(state.root);
  state = undefined;
 };

 /** Arms (or re-arms) finalization of a settled turn that OMP may still restart. */
 const armSettle = (delayMs: number, outcome: { continuationMissing?: boolean } = {}) => {
  const target = state;
  if (!target) return;
  clearTimeout(target.settleTimer);
  const timer = setTimeout(() => {
   try {
    if (state !== target || target.runsOpen > 0) return;
    finalizeRoot({ cancelled: false, ...outcome });
    if (!options.replay) void flush();
   } catch (error) {
    // A raw timer runs outside OMP's handler isolation; never let it throw.
    console.error("[pi-langfuse] failed to finalize a settled turn:", error);
   }
  }, delayMs);
  timer.unref?.();
  target.settleTimer = timer;
 };

 /** Point-in-time observation under the open tool call, or the turn. */
 const emitTurnEvent = (
  name: string,
  attributes: { input?: unknown; output?: unknown; level?: "WARNING" | "ERROR"; statusMessage?: string; metadata?: Record<string, unknown> },
  toolCallId?: string,
 ) => {
  if (!state) return;
  const parent = (toolCallId ? state.openTools.get(toolCallId)?.obs : undefined) ?? state.root;
  observe(name, () => startObservation(name, observationAttributes(attributes), { asType: "event", parentSpanContext: parent.otelSpan.spanContext(), startTime: now() }));
 };

 if (options.sdk && !options.replay) {
  registerSdkCompactionUsage(options.sdk,
   () => ensureRuntime(config).provider.getTracer("omp-compaction-usage"),
   () => state?.root.otelSpan.spanContext(),
   (event, ctx, parent) => {
    sdkCompactionCalls++;
    emitSummarizationGeneration(COMPACTION_OBSERVATION_NAME, ctx, {
     model: event.model || undefined,
     usageDetails: event.usage ? buildUsageDetails(event.usage) : undefined,
     costDetails: event.costDetails,
     metadata: {
      source: "sdk_oneshot_usage", sdk_oneshot_kind: event.kind,
      provider: event.provider, usage_missing: !event.usage,
      input_missing: true, output_missing: true,
      ...(compactionReason ? { compaction_reason: compactionReason } : {}),
     },
    }, { startedAt: event.startedAt, parentAtStart: parent ?? null, endedAt: event.endedAt });
   });
 }

 pi.on("session_start", async (_event, ctx) => {
  if (!options.replay) activeTracingSessions.add(sessionToken);
  debug("session_start", ctx.agent.id);
  if (ctx.hasUI) ctx.ui?.setStatus("langfuse", "langfuse ✓");
  if (ctx.agent.kind !== "main" || !pi.exec) return;
  try {
   const result = await pi.exec("git", ["rev-parse", "--abbrev-ref", "HEAD"], { timeout: 1000 });
   gitBranch = result.code === 0 ? result.stdout.trim() : undefined;
  } catch {
   gitBranch = undefined;
  }
 });

 const continuationTurnNumber = (ctx: TracingContext): number => {
  try {
   const users = activeEntries(ctx).filter(e => e.type === "message" && e.message?.role === "user");
   fallbackTurnCounter = Math.max(users.length, 1);
  } catch {
   fallbackTurnCounter = Math.max(fallbackTurnCounter, 1);
  }
  return fallbackTurnCounter;
 };

 // One trace per turn. A continuation is a run OMP starts without a user prompt
 // (voice/live delegation, background job results, MCP inventory notices). It fires
 // no before_agent_start, and without its own trace every generation in it was lost.
 const openTurn = (
  ctx: TracingContext,
  turn: { prompt: string; images?: unknown; systemPrompt?: string | readonly string[]; continuation: boolean },
 ): PromptState => {
  const agent = ctx.agent;
  const sessionId = ctx.sessionManager.getSessionId();
  const turnNumber = turn.continuation ? continuationTurnNumber(ctx) : resolveTurnNumber(ctx, turn.prompt);
  const promptImages = extractImages(turn.images);
  const userText = [turn.prompt, ...promptImages.map(describeImage)].filter(Boolean).join("\n");
  lastPromptText = userText;
  // A continuation's context event already captured the history of its first request.
  if (!turn.continuation) lastContextHistory = undefined;

  const inProcessParent = resolveSubagentParent(agent, registryScope);
  const crossProcessParent = agent.kind === "main" && !options.replay ? INHERITED_PARENT : undefined;
  const crossProcessSubagent = crossProcessParent?.source === "subagent";
  const parentSpanContext = inProcessParent?.spanContext ?? crossProcessParent?.spanContext;
  const title = sessionTitleOf(ctx);
  const traceAttributes = traceAttributesFor(config, {
   // A subagent joins its parent's trace and keeps the parent's trace name.
   name: inProcessParent || crossProcessSubagent ? undefined : TRACE_NAME,
   sessionId: crossProcessSubagent ? (crossProcessParent?.sessionId ?? sessionId) : sessionId,
   external: crossProcessParent?.externalTrace ?? false,
   backfill: !!options.replay,
   title,
  });
  const systemPrompt = turn.continuation
   ? readSystemPrompt(ctx)
   : readSystemPrompt({ getSystemPrompt: () => turn.systemPrompt });
  const thinkingLevel = pi.getThinkingLevel();

  const observationName = agent.kind === "main" ? MAIN_AGENT_OBSERVATION_NAME : `${agent.name ?? "omp-subagent"}-agent`;
  const root = startTraceRoot(traceAttributes, () =>
   observe(observationName, () => startObservation(observationName, observationAttributes({
    input: turn.continuation || (options.replay && !userText) ? undefined : { role: "user", content: userText },
    metadata: {
     source: "omp",
     extension: EXTENSION_NAME,
     extension_version: EXTENSION_VERSION,
     session_id: sessionId,
     ...(title ? { session_title: title } : {}),
     turn_number: turnNumber,
     ...(turn.continuation ? { turn_trigger: "continuation" } : {}),
     cwd: ctx.cwd,
     agent_id: agent.id,
     agent_name: agent.name,
     agent_kind: agent.kind,
     agent_depth: agent.depth,
     ...(agent.parentId ? { parent_agent_id: agent.parentId } : {}),
     ...(inProcessParent ? { parent_session_id: inProcessParent.parent.sessionId } : {}),
     ...(crossProcessSubagent
      ? {
       cross_process_subagent: true,
       subagent_depth: crossProcessParent?.depth,
       parent_session_id: crossProcessParent?.sessionId,
      }
      : {}),
     ...(crossProcessParent?.source === "attached" ? { attached_to_external_parent: true } : {}),
     ...(gitBranch ? { git_branch: gitBranch } : {}),
     ...(ctx.model ? { model: ctx.model.id, provider: ctx.model.provider } : {}),
     ...(thinkingLevel ? { thinking_level: thinkingLevel } : {}),
     ...(systemPrompt ? { system_prompt: systemPrompt } : {}),
    },
   }), { asType: "agent", startTime: now(), ...(parentSpanContext ? { parentSpanContext } : {}) })),
  );

  const next: PromptState = {
   agent,
   sessionId,
   registryKey: `${registryScope}${agent.id}`,
   root,
   turnNumber,
   generationCount: 0,
   openTools: new Map(),
   pendingToolResults: [],
   sawError: false,
   userText,
   turnImages: [...promptImages],
   systemPrompt,
   runsOpen: 0,
   openGenerations: [],
   awaitingRestart: false,
   steerPendingStart: false,
   interrupted: false,
   steers: 0,
   promptSequence: ++promptSequence,
  };
  liveTurns.set(next.registryKey, next);
  if (!turn.continuation) {
   try {
    const messages = ctx.sessionManager.getEntries();
    for (let index = messages.length - 1; index >= 0; index--) {
     const message = messages[index]?.message;
     if (message?.role === "user" && extractText(message.content) === turn.prompt) {
      messageOwners.set(message, next.promptSequence);
      break;
     }
    }
   } catch { debug("prompt message identity unavailable"); }
  }
  if (agent.kind === "main" && !options.replay) publishParentContext(root, sessionId);
  debug("root created", agent.id, "turn", turnNumber, turn.continuation ? "continuation" : "prompt");
  return next;
 };

 pi.on("before_agent_start", (event, ctx) => {
  try { ensureRuntime(config); }
  catch (error) {
   if (options.replay) throw error;
   console.error("[pi-langfuse]", String(error));
   return;
  }
  if (!options.replay) activeTracingSessions.add(sessionToken);
  if (state && (state.runsOpen > 0 || state.awaitingRestart)) {
   // OMP fires before_agent_start for every dequeued steer/follow-up batch,
   // including one it delivers by aborting and restarting the loop. It belongs
   // to the running turn, not a new one.
   state.steers += 1;
   const images = extractImages(event.images);
   const content = [event.prompt, ...images.map(describeImage)].filter(Boolean).join("\n");
   emitTurnEvent(STEER_OBSERVATION_NAME, {
    input: { role: "user", content },
    metadata: { steer_index: state.steers },
   });
   state.root.update({ metadata: { steer_count: state.steers } });
   state.steerPendingStart = true;
   if (state.settleTimer) armSettle(RESTART_GRACE_MS * 5);
   return;
  }
  // A superseded run's end can still arrive after the new prompt opens its turn.
  supersededRunsOpen += unclaimedRunStarts;
  unclaimedRunStarts = 0;
  // A turn still open here has no run and no restart pending: a prompt-less
  // run that never reported its start. The new prompt cuts it short.
  if (state) finalizeRoot({ cancelled: true });
  state = openTurn(ctx, { prompt: event.prompt, images: event.images, systemPrompt: event.systemPrompt, continuation: false });
 });

 pi.on("context", event => {
  if (state) {
   for (let index = event.messages.length - 1; index >= 0; index--) {
    const message = event.messages[index]!;
    if (message.role === "user") {
     messageOwners.set(message, state.promptSequence);
     break;
    }
   }
  }
  lastContextHistory = buildHistoryInput(event.messages);
  debug("context captured", lastContextHistory?.length ?? 0, "messages");
 });

 // Other extensions' before_agent_start overrides settle before the loop
 // starts; this is the system prompt the provider actually receives.
 pi.on("agent_start", (_event, ctx) => {
  if (!state) {
   unclaimedRunStarts++;
   return;
  }
  clearTimeout(state.settleTimer);
  state.settleTimer = undefined;
  state.awaitingRestart = false;
  state.steerPendingStart = false;
  state.runsOpen++;
  const systemPrompt = readSystemPrompt(ctx);
  if (!systemPrompt || systemPrompt === state.systemPrompt) return;
  state.systemPrompt = systemPrompt;
  state.root.update({ metadata: { system_prompt: systemPrompt } });
  debug("system prompt captured", systemPrompt.length);
 });

 const startGeneration = (payload: unknown, ctx: TracingContext, missingRequest = false, startedAt = now()): void => {
  if (!state) {
   // A run OMP started without a prompt still gets a trace; replay keeps archived pairing.
   if (options.replay) return;
   try { ensureRuntime(config); }
   catch (error) {
    console.error("[pi-langfuse]", String(error));
    return;
   }
   activeTracingSessions.add(sessionToken);
   state = openTurn(ctx, { prompt: "", continuation: true });
   // The run issuing this request already reported its agent_start.
   if (unclaimedRunStarts > 0) {
    unclaimedRunStarts--;
    state.runsOpen = 1;
   }
  }
  const prompt = state;
  let history = missingRequest ? undefined : lastContextHistory;
  let systemPrompt = state.systemPrompt;
  let tools = activeToolDefinitions(pi);
  let requestNormalized = false;
  let missingToolNames = 0;
  let inferredToolCallIds: string[] | undefined;
  if (!missingRequest && ctx.model) {
   try {
    const normalized = normalizeProviderRequest(payload, ctx.model.api);
    inferredToolCallIds = normalized.inferredToolCallIds;
    // Responses deltas need the context callback (or the replay runner's anchored
    // history). Full requests win over pre-provider context, including injected
    // reminders and provider-side transformations.
    if (!normalized.delta) {
     const enriched = enrichToolResults(normalized.messages, ctx.model.api);
     normalized.messages = enriched.messages;
     missingToolNames = enriched.missingNames;
     for (const message of normalized.messages) {
      if (Array.isArray(message.content) && message.content.every(part => part.type === "text")) {
       message.content = extractText(message.content);
      }
     }
     history = buildHistoryInput(normalized.messages);
     systemPrompt = normalized.systemPrompt.join("\n\n") || undefined;
    }
    tools = normalized.tools;
    requestNormalized = true;
   } catch {
    // New provider dialects still retain the host context and registry.
   }
  }
  // omp's session-title fork shares the session's provider hooks but never
  // yields a message here. Recorded as its own closed generation so it neither
  // supersedes the reply in flight nor receives that reply's output.
  if (!missingRequest && isTitleRequest(payload)) {
   const payloadModel = payload && typeof payload === "object" && "model" in payload && typeof payload.model === "string"
    ? payload.model : undefined;
   const titleInput = history
    ? systemPrompt ? [{ role: "system", content: systemPrompt }, ...history] : history
    : undefined;
   const titleObs = observe(TITLE_OBSERVATION_NAME, () => startObservation(TITLE_OBSERVATION_NAME, observationAttributes({
    input: titleInput,
    model: payloadModel ?? ctx.model?.id,
    metadata: { side_request: "session_title", usage_missing: true },
   }), { asType: "generation", parentSpanContext: prompt.root.otelSpan.spanContext(), startTime: startedAt }));
   titleObs.end(now());
   return;
  }
  // The running loop picked the steer up itself; no restart is coming for it.
  state.steerPendingStart = false;
  const modelSwitch = missingRequest ? undefined : modelSwitchFor(payload);
  if (modelSwitch) {
   emitTurnEvent(SWITCH_MODEL_OBSERVATION_NAME, {
    level: "WARNING",
    input: { from: modelLabel(modelSwitch.from), to: modelLabel(modelSwitch.to) },
   });
  }
  const index = ++state.generationCount;
  const baseInput: unknown[] =
   missingRequest ? [] : history ??
    (index === 1
     ? [{ role: "user", content: lastPromptText }]
     : state.pendingToolResults.length
      ? [{ role: "tool", tool_results: state.pendingToolResults }]
      : []);
  const generationInput = missingRequest ? undefined : systemPrompt
   ? [{ role: "system", content: systemPrompt }, ...baseInput]
   : baseInput.length ? baseInput : undefined;

  const obs = observe(GENERATION_OBSERVATION_NAME, () => startObservation(GENERATION_OBSERVATION_NAME, observationAttributes({
   input: attachToolDefinitions(generationInput, tools),
   model: ctx.model?.id,
   modelParameters: missingRequest ? undefined : extractModelParameters(payload, ctx.model, pi.getThinkingLevel()),
   metadata: {
    assistant_index: index - 1,
    input_source: history ? "context" : "delta",
    ...(requestNormalized ? { provider_payload_normalized: true } : {}),
    ...(inferredToolCallIds?.length ? { inferred_tool_call_ids: inferredToolCallIds } : {}),
    ...(missingToolNames ? { tool_result_names_missing: missingToolNames } : {}),
    ...(missingRequest ? { request_missing: true } : {}),
    ...(options.replay ? { usage_missing: true } : {}),
    ...(history ? { history_message_count: history.length } : {}),
    ...(ctx.model
     ? { provider: ctx.model.provider, api: ctx.model.api, context_window: ctx.model.contextWindow }
     : {}),
    ...(ctx.agent.id !== prompt.agent.id ? { agent_id: ctx.agent.id, agent_name: ctx.agent.name } : {}),
    ...(modelSwitch
     ? { model_switch_from: modelLabel(modelSwitch.from), model_switch_to: modelLabel(modelSwitch.to) }
     : {}),
   },
  }), { asType: "generation", parentSpanContext: prompt.root.otelSpan.spanContext(), startTime: startedAt }));
  state.openGenerations.push({ obs, index, startedAt, sawFirstToken: false, bound: false, finished: false });
 };
 pi.on("before_provider_request", (event, ctx) => startGeneration(event.payload, ctx));

 /**
  * The open generation an assistant message answers. omp stamps the message with
  * the time it sent the request, so the request that started nearest to it wins:
  * a side request (prewalk, memory recall) omp sends while the reply is pending
  * stays open for `closeUnmatched`. Without a timestamp the newest request is assumed.
  */
 const generationFor = (message: unknown): OpenGeneration | undefined => {
  if (!state) return undefined;
  const open = state.openGenerations.filter(gen => !gen.finished);
  const sentAt = message && typeof message === "object" && "timestamp" in message ? message.timestamp : undefined;
  if (typeof sentAt !== "number") return open.at(-1);
  let nearest: OpenGeneration | undefined;
  let distance = Number.POSITIVE_INFINITY;
  for (const gen of open) {
   const delta = Math.abs(gen.startedAt.getTime() - sentAt);
   // Ties go to the newer request, as before omp stamped messages.
   if (delta <= distance) {
    distance = delta;
    nearest = gen;
   }
  }
  return nearest;
 };

 /**
  * Requests no assistant message answered: side requests, and requests a retry
  * replaced. A turn's `turn_end` can arrive after the next request started, so
  * the sweep leaves requests whose reply is already streaming alone.
  */
 const closeUnmatched = (reason: "unmatched_request" | "superseded") => {
  if (!state) return;
  const keep: OpenGeneration[] = [];
  for (const gen of state.openGenerations) {
   if (gen.finished) continue;
   if (gen.bound && reason === "unmatched_request") {
    keep.push(gen);
    continue;
   }
   gen.obs.update({
    ...(reason === "superseded" ? { level: "WARNING", statusMessage: "Superseded by provider retry" } : {}),
    metadata: { [reason]: true },
   });
   gen.obs.end(now());
   gen.finished = true;
  }
  state.openGenerations = keep;
 };

 pi.on("message_start", event => {
  if (event.message.role !== "assistant") return;
  const gen = generationFor(event.message);
  if (gen) gen.bound = true;
 });

 pi.on("after_provider_response", event => {
  // The response follows its request directly; the newest open request is it.
  const gen = state?.openGenerations.filter(g => !g.finished).at(-1);
  if (!gen) return;
  gen.obs.update({
   metadata: {
    http_status: event.status,
    ...(event.requestId ? { provider_request_id: event.requestId } : {}),
   },
  });
 });

 pi.on("message_update", event => {
  const gen = generationFor(event.message);
  if (!gen) return;
  gen.bound = true;
  if (gen.sawFirstToken) return;
  const content = "content" in event.message ? event.message.content : undefined;
  if (
   extractText(content).length > 0 ||
   extractThinking(content).length > 0 ||
   extractToolCalls(content).length > 0
  ) {
   gen.sawFirstToken = true;
   gen.obs.update({ completionStartTime: now() });
  }
 });

 pi.on("turn_end", () => closeUnmatched("unmatched_request"));

 pi.on("message_end", (event, ctx) => {
  if (!state) return;
  const message = event.message;
  if (message.role !== "assistant") return;
  messageOwners.set(message, state.promptSequence);
  const usage = isUsage(message.usage) ? message.usage : undefined;
  // An archived orphan retains its output and usage without inventing input or
  // choosing one of several possible requests. Live lifecycle behavior is unchanged.
  if (options.replay && !state.openGenerations.some(gen => !gen.finished)) {
   const startedAt = new Date(now().getTime() - (message.duration ?? 0));
   startGeneration(undefined, ctx, true, startedAt);
  }
  const gen = generationFor(message);
  if (!gen) return;

  const text = extractText(message.content);
  const tools = extractToolCalls(message.content);
  const thinking = extractThinking(message.content);
  const isError = message.stopReason === "error" || message.stopReason === "aborted";
  if (message.stopReason === "error") state.sawError = true;
  const servedModel = message.upstreamModel || message.model;
  const cache = cacheAssessmentFor(message);

  gen.obs.update({
   output: {
    role: "assistant",
    ...(text ? { content: markDataUris(text) } : {}),
    ...(thinking.length ? { thinking } : {}),
    ...(tools.length ? { tool_calls: tools } : {}),
   },
   model: servedModel,
   usageDetails: usage ? buildUsageDetails(usage) : undefined,
   // In live traces the first content-bearing `message_update` reached the
   // plugin only milliseconds before `message_end`; the provider's ttft is exact.
   ...(message.ttft !== undefined && gen.startedAt.getTime() + message.ttft <= now().getTime()
    ? { completionStartTime: new Date(gen.startedAt.getTime() + message.ttft) }
    : {}),
   costDetails: usage ? buildCostDetails(usage) : undefined,
   ...(isError
    ? { level: "ERROR" as const, statusMessage: message.errorMessage || `stopReason: ${message.stopReason}` }
    : cache?.loss
     ? { level: "WARNING" as const, statusMessage: `Cache loss (${cache.classification}): ${cache.missedReusable} reusable tokens re-processed` }
     : {}),
   metadata: {
    tool_count: tools.length,
    stop_reason: message.stopReason,
    provider: message.provider,
    api: message.api,
    ...(message.responseId ? { response_id: message.responseId } : {}),
    ...(servedModel !== message.model ? { requested_model: message.model } : {}),
    ...(message.upstreamProvider ? { upstream_provider: message.upstreamProvider } : {}),
    ...(message.errorStatus !== undefined ? { error_status: message.errorStatus } : {}),
    ...(message.ttft !== undefined ? { ttft_ms: message.ttft } : {}),
    usage_missing: usage === undefined,
    ...(options.replay && usage ? { usage_source: "archived_assistant_message" } : {}),
    ...(message.duration !== undefined ? { duration_ms: message.duration } : {}),
    ...(message.disabledFeatures?.length ? { disabled_features: message.disabledFeatures } : {}),
    ...(usage ? usageMetadata(usage) : {}),
    ...(message.credentialId !== undefined ? { credential_id: message.credentialId } : {}),
    ...(cache ? cacheMetadata(cache) : {}),
   },
  });
  gen.obs.end(now());
  gen.finished = true;
  state.openGenerations = state.openGenerations.filter(open => open !== gen);
  if (text) state.lastAssistantText = text;
  state.pendingToolResults = [];
  if (cache?.accountSwitch) {
   emitTurnEvent(SWITCH_ACCOUNT_OBSERVATION_NAME, {
    level: "WARNING",
    input: { provider: cache.accountSwitch.provider, from: cache.accountSwitch.from, to: cache.accountSwitch.to },
   });
  }
 });

 pi.on("tool_execution_start", (event, ctx) => {
  if (!state) return;
  const parentSpanContext = state.root.otelSpan.spanContext();
  const redactedArgs = redactLangfuseKeys(event.args);
  const serializedArgs = safeStringify(redactedArgs);
  // See toDataUri: the media processor would upload a prefix of an inline
  // base64 blob as a corrupt file, so blobs in arguments become markers.
  const input = /data:[^;,]{0,100};base64,/.test(serializedArgs) ? markDataUris(serializedArgs) : redactedArgs;
  const attributes = {
   input,
   metadata: {
    tool_name: event.toolName,
    tool_call_id: event.toolCallId,
    ...(event.intent ? { intent: event.intent } : {}),
    ...(ctx.agent.id !== state.agent.id ? { agent_id: ctx.agent.id, agent_name: ctx.agent.name } : {}),
   },
  };
  const options = { parentSpanContext, startTime: now() };
  const obs = observe(event.toolName, () =>
   RETRIEVER_TOOLS.has(event.toolName)
    ? startObservation(event.toolName, observationAttributes(attributes), { ...options, asType: "retriever" })
    : startObservation(event.toolName, observationAttributes(attributes), { ...options, asType: "tool" }),
  );
  state.openTools.set(event.toolCallId, {
   obs,
   name: event.toolName,
   startedAt: now(),
   spawnNames: spawnedAgentNames(event.args),
  });
 });

 pi.on("tool_execution_end", (event, ctx) => {
  if (!state) return;
  const open = state.openTools.get(event.toolCallId);
  if (!open) return;
  state.openTools.delete(event.toolCallId);

  const result = event.result && typeof event.result === "object" ? event.result : undefined;
  const content = result && "content" in result ? result.content : undefined;
  const usage = result && "usage" in result && isUsage(result.usage) ? result.usage : undefined;
  const details = result && "details" in result ? result.details : undefined;
  const images = extractImages(content);
  const rawOutput = renderContentWithImageMarkers(content) || safeStringify(content);
  if (event.isError) state.sawError = true;
  state.turnImages.push(...images);

  if (usage) {
   // OMP does not report which model a tool used internally. The session
   // model keeps the no-cost fallback priceable; the flag marks the
   // attribution as approximate.
   // The observation method takes no start time; the top-level form does.
   observe(TOOL_USAGE_OBSERVATION_NAME, () => startObservation(TOOL_USAGE_OBSERVATION_NAME, observationAttributes({
    usageDetails: buildUsageDetails(usage),
    costDetails: buildCostDetails(usage),
    model: ctx.model?.id,
    metadata: {
     tool_call_id: event.toolCallId,
     tool_name: open.name,
     source: "tool_result_usage",
     model_is_session_model: true,
     ...usageMetadata(usage),
    },
   }), { asType: "generation", parentSpanContext: open.obs.otelSpan.spanContext(), startTime: open.startedAt })).end(now());
   debug("tool result usage traced", open.name, usage.input, usage.output);
  }

  open.obs.update({
   output: markDataUris(rawOutput) || undefined,
   ...(event.isError ? { level: "ERROR" as const, statusMessage: "Tool execution failed" } : {}),
   metadata: {
    is_error: event.isError,
    ...(images.length ? { image_count: images.length } : {}),
    ...(details !== undefined ? { result_details: redactToolDetails(details) } : {}),
   },
  });
  open.obs.end(now());

  state.pendingToolResults.push({ tool_call_id: event.toolCallId, name: open.name, content: rawOutput });
 });

 pi.on("tool_approval_requested", event => {
  emitTurnEvent(REQUEST_TOOL_APPROVAL_OBSERVATION_NAME, {
   input: { tool_name: event.toolName, approval_mode: event.approvalMode },
   metadata: { tool_call_id: event.toolCallId, ...(event.reason ? { reason: event.reason } : {}) },
  },
   event.toolCallId,);
 });

 pi.on("tool_approval_resolved", event => {
  emitTurnEvent(RESOLVE_TOOL_APPROVAL_OBSERVATION_NAME, {
   output: { approved: event.approved },
   ...(event.approved ? {} : { level: "WARNING" as const, statusMessage: event.reason || "Tool call denied" }),
   metadata: { tool_call_id: event.toolCallId, ...(event.reason ? { reason: event.reason } : {}) },
  },
   event.toolCallId,);
 });

 pi.on("auto_retry_start", event => {
  // The retry re-sends the request; the one it replaces never gets a message.
  closeUnmatched("superseded");
  emitTurnEvent(RETRY_REQUEST_OBSERVATION_NAME, {
   level: "WARNING",
   statusMessage: event.errorMessage,
   metadata: {
    attempt: event.attempt,
    max_attempts: event.maxAttempts,
    delay_ms: event.delayMs,
    ...(event.errorId !== undefined ? { error_id: event.errorId } : {}),
   },
  });
 });

 pi.on("auto_retry_end", event => {
  emitTurnEvent(SETTLE_RETRY_OBSERVATION_NAME, {
   output: { success: event.success },
   ...(event.success ? {} : { level: "ERROR" as const, statusMessage: event.finalError || "Retries exhausted" }),
   metadata: { attempt: event.attempt },
  });
 });

 pi.on("retry_fallback_applied", event => {
  emitTurnEvent(APPLY_MODEL_FALLBACK_OBSERVATION_NAME, {
   level: "WARNING",
   input: { from: event.from, to: event.to },
   metadata: { role: event.role, ...(event.reason ? { reason: event.reason } : {}) },
  });
 });

 pi.on("retry_fallback_succeeded", event => {
  emitTurnEvent(CONFIRM_MODEL_FALLBACK_OBSERVATION_NAME, { output: { model: event.model }, metadata: { role: event.role } });
 });

 pi.on("ttsr_triggered", event => {
  emitTurnEvent(TRIGGER_TTSR_OBSERVATION_NAME, {
   level: "WARNING",
   input: { rules: event.rules.map(rule => rule.name) },
   metadata: { rule_paths: event.rules.map(rule => rule.path) },
  });
 });

 pi.on("auto_compaction_start", event => {
  compactionStartedAt ??= now();
  compactionReason = event.reason;
  sdkCompactionCalls = 0;
  emitTurnEvent(START_AUTO_COMPACTION_OBSERVATION_NAME, { metadata: { reason: event.reason, action: event.action } });
 });

 pi.on("auto_compaction_end", event => {
  emitTurnEvent(END_AUTO_COMPACTION_OBSERVATION_NAME, {
   ...(event.errorMessage ? { level: "ERROR" as const, statusMessage: event.errorMessage } : {}),
   metadata: {
    action: event.action,
    aborted: event.aborted,
    will_retry: event.willRetry,
    ...(event.skipped ? { skipped: true } : {}),
   },
  });
 });

 const emitSummarizationGeneration = (
  name: string,
  ctx: TracingContext,
  attributes: MaintenanceAttributes,
  timing: MaintenanceOptions = {},
 ): void => {
  const startedAt = timing.startedAt ?? now();
  const endedAt = timing.endedAt ?? now();
  const parentSpanContext = timing.parentAtStart === undefined
   ? state?.root.otelSpan.spanContext() : timing.parentAtStart ?? undefined;
  const start = (values: MaintenanceAttributes, parent?: SpanContext) => {
   const sdkOptions = { startTime: startedAt, ...(parent ? { parentSpanContext: parent } : {}) };
   return observe(name, () => timing.asType === "event"
    ? startObservation(name, observationAttributes(values), { ...sdkOptions, asType: "event" })
    : startObservation(name, observationAttributes(values), { ...sdkOptions, asType: "generation" }));
  };
  if (parentSpanContext) {
   start(attributes, parentSpanContext).end(endedAt);
   return;
  }
  // Idle maintenance (`/compact`, tree navigation) runs outside any turn
  // and becomes its own trace in the session.
  ensureRuntime(config);
  const sessionId = ctx.sessionManager.getSessionId();
  const crossProcessParent = ctx.agent.kind === "main" && !options.replay ? INHERITED_PARENT : undefined;
  startTraceRoot(
   traceAttributesFor(config, {
    name: crossProcessParent ? undefined : name,
    sessionId: crossProcessParent?.sessionId ?? sessionId,
    external: crossProcessParent?.externalTrace ?? false,
    backfill: !!options.replay,
   }),
   () =>
    start({
     ...attributes,
     metadata: {
      ...attributes.metadata,
      source: "omp",
      extension: EXTENSION_NAME,
      extension_version: EXTENSION_VERSION,
      session_id: sessionId,
      agent_id: ctx.agent.id,
     },
    }, crossProcessParent?.spanContext),
  ).end(endedAt);
  if (!options.replay) void flush();
 };

 pi.on("session_before_compact", () => {
  if (!compactionStartedAt) compactionReason = undefined;
  compactionStartedAt ??= now();
  sdkCompactionCalls = 0;
 });

 // The committed summary is a result event when real SDK calls supplied billing;
 // archived/older hosts retain an explicitly usage-missing summary generation.
 pi.on("session_compact", (event, ctx) => {
  const startedAt = compactionStartedAt;
  compactionStartedAt = undefined;
  const consumedCalls = sdkCompactionCalls;
  sdkCompactionCalls = 0;
  const entry = event.compactionEntry;
  emitSummarizationGeneration(
   consumedCalls ? COMPACTION_RESULT_OBSERVATION_NAME : COMPACTION_OBSERVATION_NAME,
   ctx,
   {
    output: entry.summary ? { role: "assistant", content: entry.summary } : undefined,
    metadata: {
     from_extension: event.fromExtension,
     ...(!consumedCalls ? { usage_missing: true } : {}),
     tokens_before: entry.tokensBefore,
     ...(entry.tokensAfter !== undefined ? { tokens_after: entry.tokensAfter } : {}),
     ...(entry.method ? { method: entry.method } : {}),
     ...(entry.shortSummary ? { short_summary: entry.shortSummary } : {}),
     ...(entry.warning ? { warning: entry.warning } : {}),
     ...(ctx.model ? { session_model: ctx.model.id, provider: ctx.model.provider } : {}),
    },
   },
   { startedAt, asType: consumedCalls ? "event" : "generation" },
  );
  debug("compaction traced", entry.tokensBefore, entry.tokensAfter);
 });

 pi.on("session_tree", (event, ctx) => {
  const entry = event.summaryEntry;
  if (!entry) return; // plain navigation without a summarization call
  emitSummarizationGeneration(BRANCH_SUMMARY_OBSERVATION_NAME, ctx, {
   output: entry.summary ? { role: "assistant", content: entry.summary } : undefined,
   metadata: {
    from_entry_id: entry.fromId,
    ...(event.fromExtension ? { from_extension: true } : {}),
    ...(ctx.model ? { session_model: ctx.model.id, provider: ctx.model.provider } : {}),
   },
  });
  debug("branch summary traced", entry.fromId);
 });

 // OMP settles a prompt with `agent_end`; `willContinue` marks an end that an
 // automatic continuation (retry, empty-stop recovery) is about to resume.
 pi.on("agent_end", async (event, ctx) => {
  let owner: number | undefined;
  for (let index = event.messages.length - 1; index >= 0; index--) {
   owner = messageOwners.get(event.messages[index]!);
   if (owner !== undefined) break;
  }
  if (!state) {
   // A run that never opened a turn, or a superseded run ending late.
   if (owner === undefined && unclaimedRunStarts > 0) unclaimedRunStarts--;
   else supersededRunsOpen = Math.max(0, supersededRunsOpen - 1);
   return;
  }
  if (owner !== undefined && owner !== state.promptSequence) {
   supersededRunsOpen = Math.max(0, supersededRunsOpen - 1);
   return;
  }
  // OMP clones message_end notifications, so only a prompt's own user message
  // resolves by identity; an end without one settles a superseded run first.
  if (owner === undefined && supersededRunsOpen > 0) {
   supersededRunsOpen--;
   state.root.update({ metadata: { agent_end_identity_missing: true } });
   debug("unidentified end settled a superseded run; preserving the active root");
   return;
  }
  state.runsOpen = Math.max(0, state.runsOpen - 1);
  const last = event.messages.at(-1);
  const aborted = !!last && last.role === "assistant" && last.stopReason === "aborted";
  if (aborted) state.interrupted = true;
  if (event.willContinue) {
   // The next agent_start rejoins this turn; if the continuation never comes
   // (it failed silently), the turn still settles.
   state.awaitingRestart = true;
   if (state.runsOpen === 0) armSettle(CONTINUATION_GRACE_MS, { continuationMissing: true });
   return;
  }
  // Another loop of this prompt (a steer's restart) is still running and owns the settle.
  if (state.runsOpen > 0) return;
  if (aborted || state.steerPendingStart) {
   // An interrupt, or OMP aborting to deliver a steer and restarting. Keep the
   // turn briefly so the restart lands in the same trace.
   state.awaitingRestart = true;
   armSettle(RESTART_GRACE_MS);
   return;
  }
  finalizeRoot({ cancelled: false });
  if (!options.replay) await flush();
  if (ctx.hasUI) ctx.ui?.setStatus("langfuse", "langfuse ✓ (trace sent)");
 });

 pi.on("session_shutdown", async (_event, ctx) => {
  if (state) finalizeRoot({ cancelled: true });
  supersededRunsOpen = 0;
  unclaimedRunStarts = 0;
  messageOwners = new WeakMap();
  activeTracingSessions.delete(sessionToken);
  if (options.replay) return;
  const exitBudget = sessionHadImages ? EXIT_FLUSH_WITH_MEDIA_TIMEOUT_MS : FLUSH_TIMEOUT_MS;
  await flush(exitBudget);
  // A main session can close before its in-process children. The last live
  // owner, not a particular agent kind, is responsible for final shutdown.
  try {
   await withTimeout(shutdownTracing(), exitBudget);
  } catch {
   // Tracing must never break shutdown.
  }
 });
}

function safeStringify(value: unknown): string {
 if (value === undefined || value === null) return "";
 if (typeof value === "string") return value;
 try {
  return JSON.stringify(value);
 } catch {
  return String(value);
 }
}
