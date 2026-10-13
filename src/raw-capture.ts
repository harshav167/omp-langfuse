import { randomUUID } from "node:crypto";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { Model, ProviderResponseMetadata, RawSseEvent } from "@oh-my-pi/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import { lookup } from "@oh-my-pi/pi-coding-agent/config/registry";
import type { RawSseDebugBuffer, RawSseDebugRecord } from "@oh-my-pi/pi-tui/apps/debug/raw-sse-buffer";
import {
 analyzePayload,
 analyzePayloadOnce,
 atomicWrite,
 compareShapes,
 digest,
 type ObjectRecord,
 object,
 perCall,
 PRUNE_INTERVAL_MS,
 pruneCaptures,
 type RequestShape,
 segment,
 RETENTION_MS,
 symbols,
} from "./raw-capture-core.ts";
import { cacheAssessmentFor, modelSwitchFor, sessionTitleOf } from "./cache-accounting.ts";

/**
 * Private local capture of every provider request, response, stream frame, and assistant message,
 * independent of Langfuse. Rolling 72-hour retention; never modifies a payload.
 */
const runtimeId = `${process.pid}-${randomUUID()}`;
const runtime = {
 id: runtimeId,
 execPath: process.execPath,
 main: process.argv[1],
 profile: process.env.OMP_PROFILE ?? process.env.PI_PROFILE ?? "default",
 pid: process.pid,
 bun: process.versions.bun,
 /** Host version, filled from `pi.pi.VERSION` when the factory runs. */
 version: "",
};
const MAX_BUFFER_BYTES = 64 * 1024 * 1024;
const RETENTION_HOURS = RETENTION_MS / 3_600_000;
const MAX_PENDING_ATTEMPTS = 256;
const MAX_ASSOCIATIONS = 4096;
const MAX_PREFIXES = 256;
const MAX_ATTEMPT_AGE_MS = 15 * 60 * 1000;
const SWEEP_INTERVAL_MS = 500;
const STREAM_FLUSH_BYTES = 1024 * 1024;
const SETTINGS = [
 "providers.cacheRetention",
 "provider.appendOnlyContext",
 "providers.anthropic.thinkingDisplay",
 "providers.anthropic.serverSideFallback",
 "providers.openaiWebsockets",
 "providers.openai-codex.historyNotes",
 "extendedContext",
 "compaction.methodOrder",
 "compaction.thresholdPercent",
 "compaction.midTurnEnabled",
 "memory.backend",
 "mnemopi.autoRecall",
 "mnemopi.injectionTokenLimit",
 "textVerbosity",
 "externalThinking",
 "tools.xdevDocs",
 "inlineToolDescriptors",
 "disabledExtensions",
] as const;
const SECRET_HEADERS =
 /(authorization|api[-_]?key|cookie|token|secret|session|(?:^|[-_])key(?:$|[-_]))/i;
let api: ExtensionAPI;
let captureRoot: string;
let enabledSse = true;
let privateRoot: Promise<void> | undefined;
let sequence = 0;
let bufferedBytes = 0;
let streamBytes = 0;
let droppedWrites = 0;
let lastWarning = 0;
let pruning: Promise<unknown> | undefined;
/** One retention timer per process, started with the first session and stopped with the last. */
let pruneTimer: NodeJS.Timeout | undefined;
let writeChain: Promise<void> = Promise.resolve();
interface Attempt {
 id: string;
 sessionId: string;
 provider: string;
 model?: string;
 startedAt: string;
 prewarm: boolean;
 /** Message hashes of the request, to recognize a later re-send of the same history. */
 history?: string[];
}
interface Correlation {
 method: "single-flight" | "response-id" | "ambiguous" | "unmatched";
 captureId?: string;
 candidates: string[];
}
interface Mirror {
 sessionId: string;
 buffer: RawSseDebugBuffer;
 lastSeq: number;
 direct: boolean;
 active: boolean;
 detach: () => void;
 requestId?: string;
 provider?: string;
 model?: string;
}
const attempts = new Map<string, Attempt[]>();
const responseLinks = new Map<string, Correlation>();
const messageLinks = new Map<string, Correlation>();
const prefixes = new Map<string, { shape: RequestShape; captureId: string; ts: string }>();
const snapshots = new Map<string, string>();
const mirrors = new Map<string, Mirror>();
const streamBatches = new Map<string, { text: string[]; bytes: number }>();
const timers = new Set<string>();
/** Weak tombstones prevent re-mirroring closed host sessions without retaining their IDs forever. */
const closedSessions = new WeakSet<object>();
function warn(message: string, details: ObjectRecord): void {
 if (Date.now() - lastWarning < 60_000) return;
 lastWarning = Date.now();
 api.logger.warn(message, details);
}
function safely(handler: () => void): void {
 try {
  handler();
 } catch (error) {
  droppedWrites++;
  warn("raw-capture: capture failed", { error: String(error) });
 }
}
function refFor(sessionId: string) {
 return api.pi.AgentRegistry.global()
  .list()
  .find(ref => ref.session?.sessionManager.getSessionId() === sessionId);
}
function identity(ctx: ExtensionContext): ObjectRecord {
 return {
  sessionId: ctx.sessionManager.getSessionId(),
  sessionTitle: sessionTitleOf(ctx),
  cwd: ctx.cwd,
  runtime,
  agent: { id: ctx.agent.id, parentId: ctx.agent.parentId, kind: ctx.agent.kind },
  model: ctx.model ? { provider: ctx.model.provider, id: ctx.model.id, api: ctx.model.api } : undefined,
 };
}
function safeUrl(value: string): string {
 try {
  const url = new URL(value);
  url.username = "";
  url.password = "";
  url.search = "";
  url.hash = "";
  return url.toString();
 } catch {
  return "<invalid URL>";
 }
}
function prepareRoot(): Promise<void> {
 privateRoot ??= (async () => {
  await fs.mkdir(captureRoot, { recursive: true, mode: 0o700 });
  const stat = await fs.lstat(captureRoot);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error("Raw capture root must be a real directory");
  await fs.chmod(captureRoot, 0o700);
 })();
 return privateRoot;
}

function setBounded<T>(map: Map<string, T>, id: string, value: T, maximum: number): void {
 map.delete(id);
 map.set(id, value);
 if (map.size > maximum) {
  const oldest = map.keys().next().value;
  if (oldest !== undefined) map.delete(oldest);
 }
}

function setAttempts(bucket: string, pending: Attempt[]): void {
 if (pending.length) attempts.set(bucket, pending);
 else attempts.delete(bucket);
}

function retireAttempts(ctx: ExtensionContext, reason: string): void {
 const sessionId = ctx.sessionManager.getSessionId();
 for (const [bucket, pending] of attempts) {
  if (!bucket.startsWith(`${sessionId}|`)) continue;
  const retired = pending.filter(attempt => !attempt.prewarm);
  if (!retired.length) continue;
  setAttempts(bucket, pending.filter(attempt => attempt.prewarm));
  record("gap", retired[0]!.provider, {
   sessionId, reason, captureIds: retired.map(attempt => attempt.id),
  });
 }
}
function enqueue(bytes: number, operation: () => Promise<void>): void {
 if (bufferedBytes + bytes > MAX_BUFFER_BYTES) {
  droppedWrites++;
  warn("raw-capture: queue overflow", { droppedWrites });
  return;
 }
 bufferedBytes += bytes;
 writeChain = writeChain
  .then(async () => { await prepareRoot(); await operation(); })
  .catch(error => {
   droppedWrites++;
   warn("raw-capture: write failed", { error: String(error) });
  })
  .finally(() => {
   bufferedBytes -= bytes;
  });
}
function record(kind: string, provider: string, data: ObjectRecord, ts = new Date().toISOString()): void {
 const eventSequence = ++sequence;
 const file = path.join(
  captureRoot,
  segment(provider),
  ts.slice(0, 10),
  `${ts.slice(11, 23).replaceAll(":", "")}-${runtimeId}-${String(eventSequence).padStart(6, "0")}-${kind}.json`,
 );
 const text = `${JSON.stringify({ schemaVersion: 2, kind, ts, eventSequence, runtime, ...data })}\n`;
 enqueue(Buffer.byteLength(text), () => atomicWrite(file, text));
}
function key(sessionId: string, provider: string, model?: string): string {
 return `${sessionId}|${provider}|${model ?? ""}`;
}
function correlation(sessionId: string, provider: string, model?: string): Correlation {
 const bucket = key(sessionId, provider, model);
 const pending = attempts.get(bucket) ?? [];
 const cutoff = Date.now() - MAX_ATTEMPT_AGE_MS;
 const expired = pending.filter(attempt => Date.parse(attempt.startedAt) < cutoff);
 if (expired.length) {
  setAttempts(bucket, pending.filter(attempt => Date.parse(attempt.startedAt) >= cutoff));
  record("gap", provider, { sessionId, reason: "request-correlation-expired", captureIds: expired.map(attempt => attempt.id) });
 }
 const candidates = (attempts.get(bucket) ?? []).map(attempt => attempt.id);
 return {
  method: candidates.length === 1 ? "single-flight" : candidates.length ? "ambiguous" : "unmatched",
  captureId: candidates.length === 1 ? candidates[0] : undefined,
  candidates,
 };
}
function finishAttempt(sessionId: string, provider: string, model: string | undefined, captureId: string): void {
 const exact = key(sessionId, provider, model);
 const pending = attempts.get(exact);
 if (pending?.some(attempt => attempt.id === captureId)) {
  setAttempts(exact, pending.filter(attempt => attempt.id !== captureId));
  return;
 }
 for (const [bucket, candidates] of attempts) {
  if (bucket.startsWith(`${sessionId}|`) && candidates.some(attempt => attempt.id === captureId)) {
   setAttempts(bucket, candidates.filter(attempt => attempt.id !== captureId));
   return;
  }
 }
}
/**
 * OMP sends one request at a time per session. An older unanswered request whose whole
 * history is still the start of this one can no longer be answered into this conversation:
 * it failed, was aborted or was re-sent (for example on another account after a 429, which
 * never reaches extensions). Retiring it keeps later replies matched one-to-one.
 */
function supersedeUnanswered(sessionId: string, provider: string, shape: RequestShape, captureId: string, ts: string): void {
 const history = shape.items.map(item => item.hash);
 const retired: Attempt[] = [];
 for (const [bucket, pending] of attempts) {
  if (!bucket.startsWith(`${sessionId}|`)) continue;
  const kept = pending.filter(attempt => {
   const contained =
    !attempt.prewarm &&
    !!attempt.history &&
    attempt.history.length <= history.length &&
    attempt.history.every((hash, index) => hash === history[index]);
   if (contained) retired.push(attempt);
   return !contained;
  });
  if (kept.length !== pending.length) setAttempts(bucket, kept);
 }
 if (!retired.length) return;
 record(
  "gap",
  provider,
  {
   sessionId,
   reason: "attempt-superseded-without-response",
   captureIds: retired.map(attempt => attempt.id),
   supersededBy: captureId,
   elapsedMs: retired.map(attempt => Date.parse(ts) - Date.parse(attempt.startedAt)),
  },
  ts,
 );
}
function snapshot(ctx: ExtensionContext): string {
 const sessionId = ctx.sessionManager.getSessionId(),
  settings = refFor(sessionId)?.session?.settings;
 const observedPrompt: unknown = ctx.getSystemPrompt();
 let prompt: string[] = [];
 if (Array.isArray(observedPrompt)) prompt = observedPrompt.filter((text): text is string => typeof text === "string");
 else if (typeof observedPrompt === "string") prompt = [observedPrompt];
 // Settings handles replaced string-path `settings.get` in 18.3; read each through the registry.
 const scope = settings ?? api.pi.settings;
 const config = Object.fromEntries(
  SETTINGS.map(name => {
   const handle = lookup(name);
   if (!handle) return [name, { unavailable: "unknown setting id" }];
   try {
    return [name, handle.get(scope)];
   } catch (error) {
    return [name, { unavailable: String(error) }];
   }
  }),
 );
 const state = {
  config,
  systemSegments: prompt.map((text, index) => ({
   index,
   hash: digest(text),
   chars: text.length,
   memoryOffset: text.indexOf("<memories>"),
  })),
  model: ctx.model
   ? {
    provider: ctx.model.provider,
    id: ctx.model.id,
    api: ctx.model.api,
    baseUrl: safeUrl(ctx.model.baseUrl),
    compat: ctx.model.compat,
    contextWindow: ctx.model.contextWindow,
   }
   : undefined,
 };
 const id = digest(state);
 if (snapshots.get(sessionId) !== id) {
  record("snapshot", ctx.model?.provider ?? "unknown", {
   ...identity(ctx),
   snapshotId: id,
   previousSnapshotId: snapshots.get(sessionId),
   state,
   capture: {
    retentionHours: RETENTION_HOURS,
    streamEnabled: enabledSse,
    maxQueuedBytes: MAX_BUFFER_BYTES,
    limitations: [
     "HTTP hook snapshots precede later request hooks; wire_request frames are serialized WebSocket observations",
     "Request headers and transport-attempt IDs are not exposed by the payload hook",
     "New raw events are captured before UI truncation; pre-attachment backlog can be incomplete",
     "Symbols lost before the hook cannot be reconstructed",
    ],
   },
  });
  snapshots.set(sessionId, id);
 }
 return id;
}
function sourceProvenance(sessionId: string): unknown {
 return refFor(sessionId)?.session?.agent.state.messages.map((message, index) => ({
  index,
  role: message.role,
  symbolKeys: symbols(message),
  perCall: perCall(message),
  contentSymbolKeys: "content" in message && Array.isArray(message.content) ? message.content.map(symbols) : [],
 }));
}
function maintenance(): Promise<unknown> {
 if (pruning) return pruning;
 pruning = prepareRoot().then(() => pruneCaptures(captureRoot))
  .then(result => {
   record("retention", "_logger", { result, retentionHours: RETENTION_HOURS });
   return result;
  })
  .catch(error => {
   warn("raw-capture: retention failed", { error: String(error) });
  })
  .finally(() => {
   pruning = undefined;
  });
 return pruning;
}
function flushStreams(): void {
 for (const [file, batch] of streamBatches) {
  streamBatches.delete(file);
  streamBytes -= batch.bytes;
  const text = batch.text.join("");
  enqueue(batch.bytes, async () => {
   await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
   await fs.appendFile(file, text, { mode: 0o600 });
  });
 }
 if (droppedWrites) {
  const count = droppedWrites;
  droppedWrites = 0;
  record("gap", "_logger", { reason: "capture-write-loss", records: count });
 }
}
function streamRecord(mirror: Mirror, entry: RawSseDebugRecord): void {
 if (!mirror.active) return;
 const provider = entry.provider ?? mirror.provider ?? "unknown",
  model = entry.model ?? mirror.model;
 const ts = new Date(entry.timestamp).toISOString();
 const outbound = entry.kind === "event" && entry.raw.some(line => line.startsWith(": ws \u2192"));
 if (outbound) mirror.requestId = undefined;
 if (entry.kind === "response") {
  mirror.requestId = entry.requestId ?? undefined;
  mirror.provider = provider;
  mirror.model = model;
 }
 const association =
  (mirror.requestId ? responseLinks.get(`${mirror.sessionId}|${mirror.requestId}`) : undefined) ??
  correlation(mirror.sessionId, provider, model);
 if (outbound && entry.kind === "event" && !entry.truncated) {
  const data = entry.raw
   .filter(line => line.startsWith("data:"))
   .map(line => line.slice(5).trimStart())
   .join("\n");
  try {
   const payload: unknown = JSON.parse(data);
   record(
    "wire_request",
    provider,
    {
     sessionId: mirror.sessionId,
     model: { provider, id: model, api: entry.api },
     correlation: association,
     transport: "websocket",
     phase: "serialized-before-send",
     payload,
     serializedPayload: data,
     analysis: { shape: analyzePayload(payload) },
    },
    ts,
   );
  } catch (error) {
   record(
    "gap",
    provider,
    { sessionId: mirror.sessionId, reason: "outbound-frame-parse-failed", error: String(error) },
    ts,
   );
  }
 }
 if (
  entry.kind === "event" &&
  !entry.truncated &&
  ["message_start", "response.created", "response.completed", "response.done"].includes(entry.event ?? "")
 ) {
  for (const line of entry.raw) {
   if (!line.startsWith("data:")) continue;
   try {
    const event = object(JSON.parse(line.slice(5)));
    const id = object(event?.message)?.id ?? object(event?.response)?.id;
    if (typeof id === "string") setBounded(messageLinks, `${mirror.sessionId}|${id}`, association, MAX_ASSOCIATIONS);
   } catch {
    /* Raw data stays available. */
   }
  }
 }
 const text = `${JSON.stringify({ schemaVersion: 2, kind: "stream", ts, runtime, sessionId: mirror.sessionId, provider, model, requestId: mirror.requestId, correlation: association, associationBasis: "provider-event-order", direction: outbound ? "outbound" : "inbound", captureStage: mirror.direct ? "before-debug-buffer-truncation" : "debug-buffer-backlog", record: entry })}\n`;
 const bytes = Buffer.byteLength(text);
 if (streamBytes + bytes > MAX_BUFFER_BYTES) {
  droppedWrites++;
  return;
 }
 const file = path.join(
  captureRoot,
  segment(provider),
  ts.slice(0, 10),
  `${ts.slice(11, 16).replace(":", "")}-${runtimeId}-${segment(mirror.sessionId)}-stream.jsonl`,
 );
 let batch = streamBatches.get(file);
 if (!batch) {
  batch = { text: [], bytes: 0 };
  streamBatches.set(file, batch);
 }
 batch.text.push(text);
 batch.bytes += bytes;
 streamBytes += bytes;
 if (entry.kind === "event" && entry.truncated)
  record(
   "gap",
   provider,
   {
    sessionId: mirror.sessionId,
    reason: "stream-event-truncated",
    originalChars: entry.originalChars,
    streamSequence: entry.sequence,
   },
   ts,
  );
 if (streamBytes >= STREAM_FLUSH_BYTES) flushStreams();
}
function sweep(): void {
 if (!enabledSse) return;
 const live = new Set<string>();
 for (const ref of api.pi.AgentRegistry.global().list()) {
  if (!ref.session) continue;
  const sessionId = ref.session.sessionManager.getSessionId();
  if (closedSessions.has(ref.session)) continue;
  live.add(sessionId);
  const buffer = ref.session.rawSseDebugBuffer,
   old = mirrors.get(sessionId);
  if (old?.buffer === buffer) continue;
  old?.detach();
  const mirror: Mirror = { sessionId, buffer, lastSeq: 0, direct: false, active: true, detach: () => { } };
  const view = buffer.snapshot();
  for (const entry of view.records) {
   if (entry.sequence > mirror.lastSeq + 1)
    record("gap", entry.provider ?? "unknown", {
     sessionId,
     reason: "stream-buffer-gap",
     afterSequence: mirror.lastSeq,
     beforeSequence: entry.sequence,
    });
   mirror.lastSeq = entry.sequence;
   streamRecord(mirror, entry);
  }
  const originalEvent = buffer.recordEvent,
   originalResponse = buffer.recordResponse;
  const eventObserver = (event: RawSseEvent, model?: Model): void => {
   safely(() =>
    streamRecord(mirror, {
     kind: "event",
     sequence: ++mirror.lastSeq,
     timestamp: Date.now(),
     provider: model?.provider,
     model: model?.id,
     api: model?.api,
     event: event.event,
     raw: event.raw,
     truncated: false,
     originalChars: event.raw.reduce((n, line) => n + line.length + 1, 0),
    }),
   );
   originalEvent.call(buffer, event, model);
  };
  const responseObserver = (response: ProviderResponseMetadata, model?: Model): void => {
   safely(() =>
    streamRecord(mirror, {
     kind: "response",
     sequence: ++mirror.lastSeq,
     timestamp: Date.now(),
     provider: model?.provider,
     model: model?.id,
     api: model?.api,
     status: response.status,
     requestId: response.requestId,
     transport:
      typeof response.metadata?.lastTransport === "string" ? response.metadata.lastTransport : undefined,
    }),
   );
   originalResponse.call(buffer, response, model);
  };
  buffer.recordEvent = eventObserver;
  buffer.recordResponse = responseObserver;
  mirror.direct = true;
  mirror.detach = () => {
   mirror.active = false;
   if (buffer.recordEvent === eventObserver) buffer.recordEvent = originalEvent;
   if (buffer.recordResponse === responseObserver) buffer.recordResponse = originalResponse;
  };
  mirrors.set(sessionId, mirror);
 }
 for (const [sessionId, mirror] of mirrors)
  if (!live.has(sessionId)) {
   mirror.detach();
   mirrors.delete(sessionId);
  }
}
function lifecycle(ctx: ExtensionContext, event: ObjectRecord): void {
 record("lifecycle", ctx.model?.provider ?? "unknown", { ...identity(ctx), event });
}

/**
 * Registers local raw capture. Runs whether or not Langfuse is configured.
 * `OMP_RAW_REQUESTS_DIR` overrides the capture root; `OMP_RAW_REQUESTS_SSE=0` skips stream frames.
 */
export function registerRawCapture(pi: ExtensionAPI): void {
 if (process.env.OMP_RAW_REQUESTS === "0") return;
 api = pi;
 runtime.version = pi.pi.VERSION;
 const override = process.env.OMP_RAW_REQUESTS_DIR?.trim();
 // Same directory as the host's getLogsDir(): the config root (profile root) holds `logs/`.
 captureRoot = override ? path.resolve(override) : path.join(path.dirname(pi.pi.getAgentDir()), "logs", "raw-requests");
 privateRoot = undefined;
 enabledSse = process.env.OMP_RAW_REQUESTS_SSE !== "0";
 pi.on("session_start", async (_event, ctx) => {
  safely(() => {
   const session = refFor(ctx.sessionManager.getSessionId())?.session;
   if (session) closedSessions.delete(session);
   sweep();
   snapshot(ctx);
   lifecycle(ctx, { type: "session_start" });
  });
  if (!timers.has(ctx.sessionManager.getSessionId())) {
   timers.add(ctx.sessionManager.getSessionId());
   ctx.setInterval(
    () =>
     safely(() => {
      sweep();
      flushStreams();
     }),
    SWEEP_INTERVAL_MS,
   );
  }
  // Session start never waits on retention: one pass now and one timer for the whole process.
  if (!pruneTimer) {
   pruneTimer = setInterval(() => void maintenance(), PRUNE_INTERVAL_MS);
   pruneTimer.unref();
   void maintenance();
  }
  pi.logger.info("raw-capture active", { dir: captureRoot, retentionHours: RETENTION_HOURS, runtimeId });
 });
 pi.on("before_agent_start", (_event, ctx) => safely(() => retireAttempts(ctx, "request-superseded-by-prompt")));
 pi.on("before_provider_request", (event, ctx) =>
  safely(() => {
   sweep();
   const sessionId = ctx.sessionManager.getSessionId(),
    provider = ctx.model?.provider ?? "unknown",
    model = ctx.model?.id;
   const ts = new Date().toISOString(),
    captureId = `${runtimeId}:${++sequence}`,
    bucket = key(sessionId, provider, model);
   correlation(sessionId, provider, model);
   const prewarm = object(event)?.isPrewarm === true;
   const analysis: {
    shape?: RequestShape;
    prefix?: ObjectRecord;
    sourceMessages?: unknown;
    modelSwitch?: unknown;
    error?: string;
   } = {};
   let shape: RequestShape | undefined;
   try {
    shape = analyzePayloadOnce(event.payload);
   } catch (error) {
    analysis.error = String(error);
   }
   if (shape && !prewarm) supersedeUnanswered(sessionId, provider, shape, captureId, ts);
   const pending = attempts.get(bucket) ?? [];
   if (pending.length >= MAX_PENDING_ATTEMPTS) {
    record("gap", provider, {
     sessionId,
     reason: "unclosed-attempt-tracking-overflow",
     captureIds: pending.map(a => a.id),
    });
    pending.length = 0;
   }
   pending.push({ id: captureId, sessionId, provider, model, startedAt: ts, prewarm, history: shape?.items.map(item => item.hash) });
   setAttempts(bucket, pending);
   let snapshotId: string | undefined;
   try {
    if (shape) {
     const previous = prefixes.get(bucket);
     analysis.shape = shape;
     analysis.prefix = previous
      ? {
       ...compareShapes(previous.shape, shape),
       previousCaptureId: previous.captureId,
       elapsedMs: Date.parse(ts) - Date.parse(previous.ts),
      }
      : { first: true };
     setBounded(prefixes, bucket, { shape, captureId, ts }, MAX_PREFIXES);
    }
    analysis.sourceMessages = sourceProvenance(sessionId);
    snapshotId = snapshot(ctx);
   } catch (error) {
    analysis.error = String(error);
   }
   const modelSwitch = modelSwitchFor(event.payload);
   if (modelSwitch) {
    analysis.modelSwitch = modelSwitch;
    lifecycle(ctx, { type: "model_switch", ...modelSwitch });
   }
   record(
    "request",
    provider,
    {
     ...identity(ctx),
     captureId,
     snapshotId,
     phase: "extension-before-provider-request",
     analysis,
     payload: event.payload,
    },
    ts,
   );
  }),
 );
 pi.on("after_provider_response", (event, ctx) =>
  safely(() => {
   sweep();
   const sessionId = ctx.sessionManager.getSessionId(),
    provider = ctx.model?.provider ?? "unknown",
    model = ctx.model?.id;
   const link = correlation(sessionId, provider, model);
   if (event.requestId) setBounded(responseLinks, `${sessionId}|${event.requestId}`, link, MAX_ASSOCIATIONS);
   const headers =
    event.headers &&
    Object.fromEntries(
     Object.entries(event.headers).map(([name, value]) => [name, SECRET_HEADERS.test(name) ? "<redacted>" : value]),
    );
   record("response", provider, {
    ...identity(ctx),
    correlation: link,
    status: event.status,
    requestId: event.requestId,
    headers,
    metadata: event.metadata,
   });
   if (link.captureId && object(event.metadata)?.type === "codex.prewarm")
    finishAttempt(sessionId, provider, model, link.captureId);
  }),
 );
 pi.on("message_end", (event, ctx) =>
  safely(() => {
   if (event.message.role !== "assistant") return;
   sweep();
   const message = event.message,
    sessionId = ctx.sessionManager.getSessionId(),
    provider = message.provider;
   const byId = message.responseId ? messageLinks.get(`${sessionId}|${message.responseId}`) : undefined;
   const link = byId?.captureId
    ? { ...byId, method: "response-id" }
    : (byId ?? correlation(sessionId, provider, message.model));
   const usage = message.usage;
   const cache = cacheAssessmentFor(message);
   record("message", provider, {
    ...identity(ctx),
    model: { provider, id: message.model, api: message.api },
    correlation: link,
    analysis: {
     input: usage?.input,
     cacheRead: usage?.cacheRead,
     cacheWrite: usage?.cacheWrite,
     output: usage?.output,
     responseId: message.responseId,
     ttftMs: message.ttft,
     durationMs: message.duration,
     stopReason: message.stopReason,
     credentialId: message.credentialId,
     ...(cache ? { cache } : {}),
    },
    message,
   });
   if (cache?.accountSwitch) lifecycle(ctx, { type: "account_switch", ...cache.accountSwitch });
   if (cache?.loss) {
    lifecycle(ctx, {
     type: "cache_loss",
     classification: cache.classification,
     missedReusable: cache.missedReusable,
     reusable: cache.reusable,
     cacheRead: cache.cacheRead,
     idleMs: cache.idleMs,
     responseId: message.responseId,
    });
   }
   if (link.captureId) finishAttempt(sessionId, provider, message.model, link.captureId);
   if (message.responseId) {
    messageLinks.delete(`${sessionId}|${message.responseId}`);
    responseLinks.delete(`${sessionId}|${message.responseId}`);
   }
  }),
 );
 pi.on("session_compact", (event, ctx) =>
  safely(() =>
   record("compaction", ctx.model?.provider ?? "unknown", {
    ...identity(ctx),
    entry: event.compactionEntry,
    fromExtension: event.fromExtension,
   }),
  ),
 );
 // The overloads are keyed by literal event name; the loop name is one of them.
 const onLifecycle = pi.on as unknown as (
  event: string,
  handler: (event: ObjectRecord, ctx: ExtensionContext) => void,
 ) => void;
 for (const name of [
  "turn_start", "auto_retry_start", "auto_retry_end", "auto_compaction_start", "auto_compaction_end",
  "retry_fallback_applied", "retry_fallback_succeeded", "session_switch", "session_branch",
 ]) {
  onLifecycle(name, (event, ctx) => safely(() => {
   if (name === "auto_retry_start") retireAttempts(ctx, "request-superseded-by-retry");
   lifecycle(ctx, { ...event });
  }));
 }
 pi.on("tool_execution_start", () => safely(sweep));
 pi.on("tool_execution_end", () => safely(sweep));
 pi.on("session_shutdown", async (_event, ctx) => {
  const sessionId = ctx.sessionManager.getSessionId();
  safely(() => {
   const session = refFor(sessionId)?.session;
   if (session) closedSessions.add(session);
   const mirror = mirrors.get(sessionId);
   mirror?.detach();
   mirrors.delete(sessionId);
   lifecycle(ctx, {
    type: "session_shutdown",
    unresolvedRequests: [...attempts.values()].flat().filter(a => a.sessionId === sessionId),
   });
   flushStreams();
  });
  await writeChain;
  await pruning;
  await writeChain;
  snapshots.delete(sessionId);
  timers.delete(sessionId);
  if (timers.size === 0 && pruneTimer) {
   clearInterval(pruneTimer);
   pruneTimer = undefined;
  }
  for (const map of [attempts, prefixes, responseLinks, messageLinks])
   for (const k of map.keys()) if (k.startsWith(`${sessionId}|`)) map.delete(k);
 });
}
