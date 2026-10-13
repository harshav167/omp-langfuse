import * as fs from "node:fs/promises";
import { isDeepStrictEqual } from "node:util";
import * as path from "node:path";
import { getAgentDir, z } from "@oh-my-pi/pi-coding-agent";
import {
  extractImages,
  extractText,
  extractThinking,
  flushTracing,
  loadConfig,
  registerTracing,
  shutdownTracing,
  tracingEnabled,
  type LangfuseConfig,
  type TracingAPI,
  type TracingContext,
  type TracingEvent,
  type TracingHistoryMessage,
  type TracingHistoryPart,
  type PiImagePart,
  type TracingAssistantMessage,
  type TracingOptions,
  type ToolDefinitionInput,
} from "./index.ts";
import { callAlias, enrichToolResults, normalizeProviderRequest, RESPONSES_APIS, type ToolCallCandidate } from "./backfill/normalize.ts";
import { historyPartSchema, recordSchema as objectSchema, type NormalizedRequest } from "./backfill/types.ts";
import { digest } from "./raw-capture-core.ts";
import {
  enumerateCaptureFiles, peekSessionId, readCapture, snapshotCaptures,
  type CaptureFile, type CaptureSnapshot,
} from "./backfill/archive.ts";

const DEFAULT_ROOT = () => process.env.OMP_RAW_REQUESTS_DIR?.trim() ||
  path.join(path.dirname(getAgentDir()), "logs", "raw-requests");

const SUPPORTED_LIFECYCLE: ReadonlySet<string> = new Set([
  "auto_retry_start",
  "auto_retry_end",
  "auto_compaction_start",
  "auto_compaction_end",
  "retry_fallback_applied",
  "retry_fallback_succeeded",
]);
const MAX_PENDING_REQUESTS = 256;
const runtimeSchema = z.object({ id: z.string().min(1) }).passthrough();
const captureSchema = z.object({
  schemaVersion: z.number().int(),
  kind: z.string(),
  ts: z.string().refine(value => Number.isFinite(Date.parse(value))),
  eventSequence: z.number().int().positive(),
  runtime: runtimeSchema,
  sessionId: z.string().optional(),
  captureId: z.string().optional(),
}).passthrough();
const agentSchema = z.object({ id: z.string().min(1), kind: z.enum(["main", "sub"]), parentId: z.string().optional() }).passthrough();
const modelSchema = z.object({
  provider: z.string().min(1),
  id: z.string().min(1),
  api: z.string().min(1),
}).passthrough();
const requestSchema = z.object({
  captureId: z.string().min(1),
  sessionId: z.string().min(1),
  cwd: z.string(),
  agent: agentSchema,
  model: modelSchema,
  payload: z.unknown(),
}).passthrough();
const correlationSchema = z.object({
  method: z.string().optional(),
  captureId: z.string().optional(),
  candidates: z.array(z.string()).optional(),
}).passthrough();
const messageSchema = z.object({
  sessionId: z.string().min(1),
  agent: agentSchema,
  cwd: z.string(),
  message: z.object({
    role: z.literal("assistant"),
    content: z.array(historyPartSchema),
    api: z.string().min(1),
    provider: z.string().min(1),
    model: z.string().min(1),
    usage: z.object({
      input: z.number(),
      output: z.number(),
      cacheRead: z.number(),
      cacheWrite: z.number(),
    }).passthrough().optional(),
    stopReason: z.enum(["stop", "length", "toolUse", "error", "aborted"]),
    upstreamModel: z.string().optional(),
    upstreamProvider: z.string().optional(),
    responseId: z.string().optional(),
    errorMessage: z.string().optional(),
    errorStatus: z.number().optional(),
    ttft: z.number().nonnegative().optional(),
    duration: z.number().nonnegative().optional(),
    disabledFeatures: z.array(z.string()).optional(),
  }).passthrough(),
  correlation: correlationSchema.optional(),
}).passthrough();
const responseSchema = z.object({
  sessionId: z.string().min(1),
  agent: agentSchema,
  cwd: z.string(),
  correlation: correlationSchema.optional(),
  status: z.number().optional(),
  requestId: z.string().optional(),
  headers: objectSchema.optional(),
  metadata: z.unknown().optional(),
}).passthrough();
const lifecycleSchema = z.object({
  sessionId: z.string().min(1),
  agent: agentSchema,
  cwd: z.string(),
  event: z.object({ type: z.string().min(1) }).passthrough(),
}).passthrough();
const compactionSchema = z.object({
  sessionId: z.string().min(1),
  agent: agentSchema,
  cwd: z.string(),
  entry: objectSchema,
  fromExtension: z.boolean().optional(),
}).passthrough();

export interface BackfillOptions {
  root?: string;
  sessions?: readonly string[];
  from?: string;
  until?: string;
  /** Maximum number of selected provider request records to replay. */
  limit?: number;
  dryRun?: boolean;
  /** Keep the private input snapshot for backend verification or a later retry. */
  keepSnapshot?: boolean;
  /** Export only these real SDK span IDs while replaying the full source context. */
  observationIds?: readonly string[];
}

export interface BackfillSpanExpectation {
  traceId: string;
  sessionId: string;
  rootObservationId?: string;
  root: 1;
  generations: number;
  tools: number;
  sourceRequests: string[];
  incomplete: boolean;
}

export interface BackfillReport {
  root: string;
  dryRun: boolean;
  policyDisabled: boolean;
  processed: number;
  imported: number;
  wouldImport: number;
  skipped: number;
  unsupported: number;
  unmatched: number;
  incomplete: number;
  requests: number;
  warnings: number;
  rootIds: string[];
  observationIds: string[];
  snapshotRoot?: string;
  sourceSpanExpectations: BackfillSpanExpectation[];
}


interface CaptureRecord extends Record<string, unknown> {
  schemaVersion: number;
  kind: string;
  ts: string;
  eventSequence: number;
  runtime: { id: string; [key: string]: unknown };
  sessionId?: string;
  captureId?: string;
}

interface SourceProvenance {
  source_file: string;
  archived_facts: string[];
  inferred_facts: string[];
  missing_data: string[];
}

interface RequestState {
  captureId: string;
  sourceFile: string;
  api: string;
  provider: string;
  model: string;
  ts: string;
  rootTraceId?: string;
  completed: boolean;
}


interface PendingTool extends ToolCallCandidate {
  args: unknown;
  requestId: string;
  rootTraceId?: string;
}

interface ArchivedPrompt {
  prompt: string;
  images: PiImagePart[];
}
interface Correlation {
  captureId?: string;
  candidates: string[];
  ambiguous: boolean;
}

interface Actor {
  runtimeId: string;
  sessionId: string;
  namespace: string;
  cwd: string;
  agent: { id: string; kind: TracingContext["agent"]["kind"]; parentId?: string };
  model?: { provider: string; id: string; api: string };
  tools: ToolDefinitionInput[];
  entries: Array<{ type: string; message: { role: string; content: unknown } }>;
  history: TracingHistoryMessage[];
  hasHistory: boolean;
  turnPending: boolean;
  turnOpen: boolean;
  currentGenerationCaptureId?: string;
  pendingRequests: Map<string, RequestState>;
  pendingTools: Map<string, PendingTool>;
  currentTime: Date;
  currentProvenance: SourceProvenance;
  currentSeed: string;
  api?: TracingAPI;
  handlers: Map<string, Array<(event: TracingEvent, ctx: TracingContext) => void | Promise<void>>>;
  context: TracingContext;
  roots: BackfillSpanExpectation[];
  activeRoot?: BackfillSpanExpectation;
  currentSystemPrompt: string[];
  latestUser?: TracingHistoryMessage;
  latestUserText?: string;
  archivedAssistantMessages: unknown[];
  historyIncomplete: boolean;
  currentFile: CaptureFile;
}

interface RunState {
  report: BackfillReport;
  observationIds: Set<string>;
  traceIds: Set<string>;
  includeObservationIds?: ReadonlySet<string>;
  incompleteCaptures: Set<string>;
  incompleteRecords: number;
  sessions?: Set<string>;
  config?: LangfuseConfig;
  canExport: boolean;
  dryRun: boolean;
  currentLimitRequestId?: string;
  stopAfterCurrentMessage: boolean;
}

function isoTime(value: string | undefined, option: string): number | undefined {
  if (value === undefined) return undefined;
  if (!/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)$/.test(value)) {
    throw new Error(`--${option} must be an ISO timestamp with a timezone`);
  }
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) throw new Error(`--${option} is not a valid ISO timestamp`);
  return parsed;
}




function parseEnvelope(raw: unknown): CaptureRecord | undefined {
  const parsed = captureSchema.safeParse(raw);
  if (!parsed.success) return undefined;
  return parsed.data as CaptureRecord;
}

function validateRecordDetails(record: CaptureRecord): boolean {
  switch (record.kind) {
    case "request": return requestSchema.safeParse(record).success;
    case "message": return messageSchema.safeParse(record).success;
    case "response": return responseSchema.safeParse(record).success;
    case "lifecycle": return lifecycleSchema.safeParse(record).success;
    case "compaction": return compactionSchema.safeParse(record).success;
    case "snapshot": return objectSchema.safeParse(record.state).success;
    case "gap": return typeof record.reason === "string";
    case "wire_request": return Object.hasOwn(record, "payload") && !!record.payload && typeof record.payload === "object";
    default: return true;
  }
}

function parseCorrelation(record: CaptureRecord): Correlation {
  const parsed = correlationSchema.safeParse(record.correlation);
  if (!parsed.success) return { candidates: [], ambiguous: false };
  const value = parsed.data;
  const candidates = value.candidates?.filter(candidate => typeof candidate === "string") ?? [];
  return {
    captureId: typeof value.captureId === "string" && value.captureId.length > 0 ? value.captureId : undefined,
    candidates,
    ambiguous: value.method === "ambiguous" || (!value.captureId && candidates.length > 1),
  };
}




function stringField(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}





function currentProvenance(
  file: CaptureFile,
  facts: string[],
  inferred: string[] = [],
  missing: string[] = [],
): SourceProvenance {
  return {
    source_file: file.relativeFile,
    archived_facts: facts,
    inferred_facts: inferred,
    missing_data: missing,
  };
}

function createActor(record: CaptureRecord, file: CaptureFile, state: RunState, scope = ""): Actor | undefined {
  const sessionId = stringField(record.sessionId);
  const identity = agentSchema.safeParse(record.agent);
  if (!sessionId || !identity.success || typeof record.cwd !== "string") return undefined;
  const context: TracingContext = {
    get cwd() { return actor.cwd; },
    hasUI: false,
    getSystemPrompt: () => actor.currentSystemPrompt,
    get agent() { return actor.agent; },
    get model() { return actor.model; },
    sessionManager: {
      getSessionId: () => actor.sessionId,
      getEntries: () => actor.entries,
    },
  };
  const actor: Actor = {
    runtimeId: record.runtime.id,
    sessionId,
    namespace: `backfill:${record.runtime.id}${scope}`,
    cwd: record.cwd,
    agent: {
      id: identity.data.id,
      kind: identity.data.kind,
      ...(identity.data.parentId ? { parentId: identity.data.parentId } : {}),
    },
    tools: [],
    entries: [],
    history: [],
    hasHistory: false,
    turnPending: false,
    turnOpen: false,
    pendingRequests: new Map(),
    pendingTools: new Map(),
    currentTime: new Date(record.ts),
    currentProvenance: currentProvenance(file, ["session_id", "agent_identity", "cwd"]),
    currentSeed: `${record.runtime.id}:${sessionId}:${record.eventSequence}:actor`,
    historyIncomplete: false,
    currentSystemPrompt: [],
    archivedAssistantMessages: [],
    currentFile: file,
    handlers: new Map(),
    context,
    roots: [],
  };
  if (state.canExport && state.config) {
    actor.api = createTracingAPI(actor);
    const replay: NonNullable<TracingOptions["replay"]> = {
      namespace: actor.namespace,
      includeObservationIds: state.includeObservationIds,
      eventId: () => actor.currentSeed,
      metadata: () => ({
        source_file: actor.currentProvenance.source_file,
        archived_facts: actor.currentProvenance.archived_facts,
        inferred_facts: actor.currentProvenance.inferred_facts,
        missing_data: actor.currentProvenance.missing_data,
        raw_capture_version: 2,
        runtime_id: actor.runtimeId,
      }),
    };
    registerTracing(actor.api, state.config, {
      now: () => actor.currentTime, replay,
      onObservation(name, context) {
        if (!state.includeObservationIds || state.includeObservationIds.has(context.spanId)) {
          state.observationIds.add(context.spanId);
          state.traceIds.add(context.traceId);
        }
        if (actor.activeRoot && actor.currentSeed.endsWith(":before-agent-start") &&
          (name === "omp-agent" || name === "omp-subagent-agent")) {
          actor.activeRoot.traceId = context.traceId;
          actor.activeRoot.rootObservationId = context.spanId;
        }
      },
    });
  }
  return actor;
}

function updateActorIdentity(actor: Actor, record: CaptureRecord, file: CaptureFile): void {
  if (typeof record.cwd === "string") actor.cwd = record.cwd;
  const identity = agentSchema.safeParse(record.agent);
  if (identity.success) {
    actor.agent = {
      id: identity.data.id,
      kind: identity.data.kind,
      ...(identity.data.parentId ? { parentId: identity.data.parentId } : {}),
    };
  }
  const model = modelSchema.safeParse(record.model);
  actor.model = model.success ? { provider: model.data.provider, id: model.data.id, api: model.data.api } : undefined;
  actor.currentTime = new Date(record.ts);
  actor.currentFile = file;
}


function createTracingAPI(actor: Actor): TracingAPI {
  return {
    on<E extends TracingEvent["type"]>(
      type: E,
      handler: (event: Extract<TracingEvent, { type: E }>, ctx: TracingContext) => void | Promise<void>,
    ): void {
      const handlers = actor.handlers.get(type) ?? [];
      handlers.push(handler as unknown as (event: TracingEvent, ctx: TracingContext) => void | Promise<void>);
      actor.handlers.set(type, handlers);
    },
    getThinkingLevel: () => undefined,
    getAllTools: () => actor.tools,
    getActiveTools: () => actor.tools.map(tool => tool.name),
  };
}

async function emit(actor: Actor, event: TracingEvent, seed: string, provenance: SourceProvenance): Promise<void> {
  actor.currentSeed = seed;
  actor.currentProvenance = provenance;
  if (!actor.api) return;
  for (const handler of actor.handlers.get(event.type) ?? []) await handler(event, actor.context);
}

function captureSeed(actor: Actor, record: CaptureRecord, eventType: string, suffix = ""): string {
  const source = stringField(record.captureId) ?? `${actor.runtimeId}:${actor.sessionId}:${record.eventSequence}`;
  return `${source}:${eventType}${suffix ? `:${suffix}` : ""}`;
}

function promptFor(user: TracingHistoryMessage | undefined): ArchivedPrompt | undefined {
  if (!user) return undefined;
  const images = extractImages(user.content);
  const prompt = extractText(user.content);
  if (!prompt && images.length === 0) return undefined;
  return { prompt, images };
}

function updateEntries(actor: Actor): void {
  actor.entries = actor.history.map(message => ({
    type: "message",
    message: { role: message.role, content: message.content },
  }));
}

function traceIdForRoot(actor: Actor, eventId: string): string {
  const name = actor.agent.kind === "main" ? "omp-agent" : "omp-subagent-agent";
  const observationSeed = `${actor.namespace}:${eventId}:${name}`;
  return digest(`omp-backfill-v1:${observationSeed}:trace`).slice(0, 32);
}

function markIncomplete(state: RunState, captureId?: string): void {
  if (captureId) state.incompleteCaptures.add(captureId);
  else state.incompleteRecords++;
}

function recordImported(state: RunState): void {
  if (state.dryRun) state.report.wouldImport++;
  else if (state.canExport) state.report.imported++;
  else state.report.skipped++;
}

async function endTurn(actor: Actor, record: CaptureRecord, file: CaptureFile, state: RunState, why: string): Promise<void> {
  if (!actor.turnOpen) return;
  const rootId = actor.activeRoot?.traceId;
  const event = {
    type: "agent_end",
    messages: actor.archivedAssistantMessages,
    willContinue: false,
  } as unknown as TracingEvent;
  const provenance = currentProvenance(
    file,
    ["archived_assistant_messages"],
    [why],
    actor.currentGenerationCaptureId ? ["agent_end_callback_not_archived"] : [],
  );
  await emit(actor, event, captureSeed(actor, record, "agent-end", why), provenance);
  if (actor.currentGenerationCaptureId) {
    const active = actor.pendingRequests.get(actor.currentGenerationCaptureId);
    if (active && !active.completed) markIncomplete(state, active.captureId);
  }
  for (const [id, tool] of actor.pendingTools) {
    if (tool.rootTraceId !== rootId) continue;
    markIncomplete(state, tool.requestId);
    actor.pendingTools.delete(id);
  }
  actor.currentGenerationCaptureId = undefined;
  actor.turnOpen = false;
  actor.activeRoot = undefined;
  actor.archivedAssistantMessages = [];
}

function toolCallParts(message: Record<string, unknown>): Array<{ id: string; name: string; args: unknown }> {
  if (!Array.isArray(message.content)) return [];
  const calls: Array<{ id: string; name: string; args: unknown }> = [];
  for (const part of message.content) {
    const parsed = objectSchema.safeParse(part);
    if (!parsed.success || parsed.data.type !== "toolCall") continue;
    const id = stringField(parsed.data.id);
    const name = stringField(parsed.data.name);
    if (id && name) calls.push({ id, name, args: parsed.data.arguments });
  }
  return calls;
}



async function finishToolResults(
  actor: Actor,
  normalized: NormalizedRequest,
  api: string,
  record: CaptureRecord,
  file: CaptureFile,
): Promise<{ unmatched: number }> {
  let unmatched = 0;
  const inferredIds = new Set(normalized.inferredToolCallIds);
  const inferredCalls = new Map<string, Extract<TracingHistoryPart, { type: "toolCall" }>>();
  for (const message of normalized.messages) {
    if (message.role !== "assistant" || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (part.type === "toolCall" && inferredIds.has(part.id)) inferredCalls.set(part.id, part);
    }
  }
  for (const result of normalized.messages) {
    if (result.role !== "toolResult") continue;
    const inferred = result.toolCallId !== undefined && inferredIds.has(result.toolCallId);
    const historyCall = result.toolCallId ? inferredCalls.get(result.toolCallId) : undefined;
    const candidates = [...actor.pendingTools.values()].filter(candidate =>
      candidate.id === result.toolCallId ||
      (RESPONSES_APIS.has(candidate.api) && RESPONSES_APIS.has(api) && callAlias(candidate.id, candidate.api) === callAlias(result.toolCallId ?? "", api)) ||
      (inferred && candidate.api === api && historyCall?.arguments !== undefined &&
        candidate.name === historyCall.name && isDeepStrictEqual(candidate.args, historyCall.arguments)),
    );
    if (candidates.length === 0) continue;
    if (candidates.length > 1) {
      unmatched++;
      continue;
    }
    const call = candidates[0]!;
    const provenance = currentProvenance(
      file,
      ["tool_result_content", ...(inferred ? [] : ["tool_result_call_id"])],
      ["tool_end_timestamp_inferred_from_following_request", ...(inferred ? ["tool_result_identity_inferred_from_google_name_and_arguments"] : [])],
      ["tool_result_usage", "tool_result_details"],
    );
    const event = {
      type: "tool_execution_end",
      toolCallId: call.id,
      toolName: call.name,
      result: { content: result.content },
      isError: result.isError ?? false,
    } as unknown as TracingEvent;
    await emit(actor, event, captureSeed(actor, record, "tool-end", call.id), provenance);
    actor.pendingTools.delete(call.id);
  }
  return { unmatched };
}



async function beginTurn(
  actor: Actor, record: CaptureRecord, file: CaptureFile,
  prompt: ArchivedPrompt | undefined, missing: string[] = [],
): Promise<void> {
  updateEntries(actor);
  const rootSeed = captureSeed(actor, record, "before-agent-start");
  const root: BackfillSpanExpectation = {
    traceId: traceIdForRoot(actor, rootSeed), sessionId: actor.sessionId,
    root: 1, generations: 0, tools: 0, sourceRequests: [],
    incomplete: !prompt || missing.length > 0,
  };
  actor.roots.push(root);
  actor.activeRoot = root;
  actor.archivedAssistantMessages = [];
  await emit(actor, {
    type: "before_agent_start", prompt: prompt?.prompt ?? "", images: prompt?.images,
    systemPrompt: actor.currentSystemPrompt,
  }, rootSeed, currentProvenance(file,
    ["session_id", "agent_identity", "cwd", ...(prompt ? ["provider_request_user_history"] : [])],
    ["prompt_boundary_reconstructed"],
    ["original_before_agent_start_event", "original_agent_end_event", "agent_name", "agent_depth", ...missing, ...(!prompt ? ["user_prompt"] : [])],
  ));
  await emit(actor, { type: "agent_start" }, captureSeed(actor, record, "agent-start"),
    currentProvenance(file, [], ["agent_start_reconstructed"], ["original_agent_start_event"]));
  actor.turnOpen = true;
  actor.turnPending = false;
}


async function processRequest(
  record: CaptureRecord,
  file: CaptureFile,
  actor: Actor,
  state: RunState,
): Promise<{ requestId?: string }> {
  const parsed = requestSchema.safeParse(record);
  if (!parsed.success) {
    state.report.unsupported++;
    markIncomplete(state, record.captureId);
    return {};
  }
  const captureId = parsed.data.captureId;
  const { provider, id: modelId, api } = parsed.data.model;
  const payload = parsed.data.payload;
  let normalized: NormalizedRequest;
  try {
    normalized = normalizeProviderRequest(payload, api);
  } catch {
    state.report.unsupported++;
    state.report.warnings++;
    markIncomplete(state, captureId);
    return { requestId: captureId };
  }
  state.report.warnings += normalized.warnings.length;
  if (normalized.warnings.length) markIncomplete(state, captureId);

  actor.tools = normalized.tools;
  actor.currentSystemPrompt = normalized.systemPrompt;
  const previousLatestText = actor.latestUserText;
  const hadHistory = actor.hasHistory;
  let incomingUser: TracingHistoryMessage | undefined;
  for (let index = normalized.messages.length - 1; index >= 0; index--) {
    const message = normalized.messages[index]!;
    if (message.role !== "user") continue;
    incomingUser = message;
    break;
  }

  const nameQuality = enrichToolResults(normalized.messages, api, actor.pendingTools.values());
  normalized.messages = nameQuality.messages;
  if (nameQuality.missingNames > 0) {
    markIncomplete(state, captureId);
    state.report.warnings += nameQuality.missingNames;
  }
  if (nameQuality.ambiguousNames > 0) state.report.unmatched += nameQuality.ambiguousNames;
  const toolFinish = await finishToolResults(actor, normalized, api, record, file);
  if (toolFinish.unmatched > 0) {
    state.report.unmatched += toolFinish.unmatched;
    markIncomplete(state, captureId);
  }

  if (normalized.delta) {
    if (!hadHistory) {
      actor.historyIncomplete = true;
      markIncomplete(state, captureId);
    }
    actor.history.push(...normalized.messages);
    if (incomingUser) actor.latestUser = incomingUser;
    actor.hasHistory = true;
  } else {
    actor.history = normalized.messages;
    actor.latestUser = incomingUser;
    actor.hasHistory = true;
    actor.historyIncomplete = false;
  }
  actor.latestUserText = actor.latestUser ? extractText(actor.latestUser.content) : undefined;
  const prompt = promptFor(actor.latestUser);
  const newPromptDetected = actor.latestUserText !== undefined && actor.latestUserText !== previousLatestText;
  if (actor.turnOpen && newPromptDetected && !actor.turnPending) {
    await endTurn(actor, record, file, state, "turn_boundary_inferred_from_new_user_history");
  }

  if (!actor.turnOpen) {
    const missing = !prompt ? ["prior_history"] : [];
    await beginTurn(actor, record, file, prompt, missing);
    if (missing.length) markIncomplete(state, captureId);
  }

  const request: RequestState = {
    captureId,
    sourceFile: file.relativeFile,
    api,
    provider,
    model: modelId,
    ts: record.ts,
    rootTraceId: actor.activeRoot?.traceId,
    completed: false,
  };
  actor.pendingRequests.set(captureId, request);
  if (actor.pendingRequests.size > MAX_PENDING_REQUESTS) {
    const oldest = actor.pendingRequests.values().next().value as RequestState | undefined;
    if (oldest) {
      actor.pendingRequests.delete(oldest.captureId);
      markIncomplete(state, oldest.captureId);
    }
  }
  if (actor.currentGenerationCaptureId) {
    const previous = actor.pendingRequests.get(actor.currentGenerationCaptureId);
    if (previous && !previous.completed) markIncomplete(state, previous.captureId);
  }
  actor.currentGenerationCaptureId = captureId;
  if (actor.activeRoot) {
    actor.activeRoot.generations++;
    actor.activeRoot.sourceRequests.push(captureId);
  }
  const contextEvent = { type: "context", messages: actor.history } as unknown as TracingEvent;
  await emit(actor, contextEvent, captureSeed(actor, record, "context"), currentProvenance(
    file,
    ["provider_request_payload", "model", "system_prompt", "tools", "history"],
    [
      ...(normalized.delta ? ["history_extended_from_previous_response_id_delta"] : []),
      ...(normalized.inferredToolCallIds?.length ? ["google_history_tool_identities_inferred"] : []),
    ],
    [
      ...(actor.historyIncomplete ? ["prior_delta_history"] : []),
      ...(nameQuality.missingNames
        ? [nameQuality.ambiguousNames ? "tool_result_name_ambiguous_match" : "tool_result_name_missing"]
        : []),
      ...(normalized.warnings.length ? ["provider_fields_not_represented"] : []),
    ],
  ));
  await emit(actor, { type: "before_provider_request", payload } as unknown as TracingEvent,
    captureSeed(actor, record, "generation"), currentProvenance(
      file,
      ["provider_request_payload", "provider", "api", "model", "cwd", "session_id"],
      [
        ...(normalized.delta ? ["delta_history_reconstructed"] : []),
        ...(normalized.inferredToolCallIds?.length ? ["google_history_tool_identities_inferred"] : []),
      ],
      [
        ...(actor.historyIncomplete ? ["previous_response_id_anchor_unavailable"] : []),
        ...(normalized.warnings.length ? ["provider_fields_not_represented"] : []),
      ],
    ));
  recordImported(state);
  return { requestId: captureId };
}

async function processResponse(record: CaptureRecord, file: CaptureFile, actor: Actor, state: RunState): Promise<void> {
  const parsed = responseSchema.safeParse(record);
  if (!parsed.success) {
    state.report.unsupported++;
    markIncomplete(state);
    return;
  }
  const correlation = parseCorrelation(record);
  const request = correlation.captureId ? actor.pendingRequests.get(correlation.captureId) : undefined;
  if (!request || request.captureId !== actor.currentGenerationCaptureId || correlation.ambiguous) {
    state.report.unmatched++;
    markIncomplete(state, correlation.captureId);
    return;
  }
  const event = {
    type: "after_provider_response",
    status: parsed.data.status,
    requestId: parsed.data.requestId,
    headers: parsed.data.headers,
    metadata: parsed.data.metadata,
  } as unknown as TracingEvent;
  await emit(actor, event, captureSeed(actor, record, "provider-response"), currentProvenance(
    file, ["provider_response_status", "request_id", "metadata"], [],
    parsed.data.status === undefined ? ["http_status"] : [],
  ));
  recordImported(state);
}

async function preserveUnmatchedMessage(
  record: CaptureRecord, file: CaptureFile, actor: Actor, state: RunState,
  correlation: Correlation, message: TracingAssistantMessage,
): Promise<void> {
  state.report.unmatched++;
  for (const candidate of correlation.candidates) markIncomplete(state, candidate);
  markIncomplete(state);
  // A separate trace preserves actual output/usage, not a guessed request pairing.
  const orphan = createActor(record, file, state, `:orphan:${record.eventSequence}`);
  if (!orphan) return;
  updateActorIdentity(orphan, record, file);
  const endTime = orphan.currentTime;
  orphan.currentTime = new Date(endTime.getTime() - (message.duration ?? 0));
  await beginTurn(orphan, record, file, undefined, ["provider_request", "request_correlation"]);
  orphan.currentTime = endTime;
  await emit(orphan, { type: "message_end", message },
    captureSeed(orphan, record, "orphan-generation"), currentProvenance(file,
      assistantFacts(message),
      ["independent_trace_for_uncorrelated_output", ...(message.duration === undefined ? [] : ["start_time_inferred_from_archived_duration"])],
      ["provider_request", "user_prompt", "request_correlation"],
    ));
  if (orphan.activeRoot) orphan.activeRoot.generations++;
  await endTurn(orphan, record, file, state, "orphan_message_boundary");
  actor.roots.push(...orphan.roots);
  recordImported(state);
}

function assistantFacts(message: TracingAssistantMessage): string[] {
  return [
    "assistant_message", "stop_reason", ...(message.usage ? ["usage"] : []),
    ...(extractThinking(message.content).length ? ["thinking"] : []),
    ...(message.errorMessage !== undefined || message.errorStatus !== undefined ? ["error"] : []),
    ...(message.ttft !== undefined ? ["ttft"] : []),
    ...(message.duration !== undefined ? ["duration"] : []),
  ];
}

async function processMessage(record: CaptureRecord, file: CaptureFile, actor: Actor, state: RunState): Promise<void> {
  const parsed = messageSchema.safeParse(record);
  if (!parsed.success) {
    state.report.unsupported++;
    markIncomplete(state);
    return;
  }
  const message = parsed.data.message;
  const correlation = parseCorrelation(record);
  const request = correlation.captureId ? actor.pendingRequests.get(correlation.captureId) : undefined;
  if (!request || request.captureId !== actor.currentGenerationCaptureId || correlation.ambiguous) {
    await preserveUnmatchedMessage(record, file, actor, state, correlation, message);
    return;
  }
  if (!message.usage) markIncomplete(state, request.captureId);
  const messageEvent = { type: "message_update", message } as unknown as TracingEvent;
  await emit(actor, messageEvent, captureSeed(actor, record, "message-update"), currentProvenance(
    file,
    assistantFacts(message),
    [],
    ["message_start_callback"],
  ));
  await emit(actor, { type: "message_end", message },
    captureSeed(actor, record, "message-end"), currentProvenance(
      file,
      assistantFacts(message),
      [],
      ["original_message_start_callback"],
    ));
  request.completed = true;
  actor.pendingRequests.delete(request.captureId);
  actor.currentGenerationCaptureId = undefined;
  actor.archivedAssistantMessages.push(message);
  actor.history.push({ role: "assistant", content: message.content });
  actor.hasHistory = true;
  updateEntries(actor);
  recordImported(state);

  const calls = toolCallParts(message);
  for (const call of calls) {
    const pending: PendingTool = {
      id: call.id,
      name: call.name,
      args: call.args,
      api: request.api,
      requestId: request.captureId,
      rootTraceId: request.rootTraceId,
    };
    if (actor.pendingTools.has(call.id)) {
      state.report.unmatched++;
      markIncomplete(state, request.captureId);
      continue;
    }
    actor.pendingTools.set(call.id, pending);
    if (actor.activeRoot) actor.activeRoot.tools++;
    await emit(actor, {
      type: "tool_execution_start",
      toolCallId: call.id,
      toolName: call.name,
      args: call.args,
    } as unknown as TracingEvent,
    captureSeed(actor, record, "tool-start", call.id), currentProvenance(
      file,
      ["tool_call_id", "tool_name", "tool_arguments"],
      ["tool_start_timestamp_inferred_from_assistant_message_timestamp"],
      ["tool_execution_start_callback", "tool_intent"],
    ));
  }
  if (request.captureId === state.currentLimitRequestId) state.stopAfterCurrentMessage = true;
}

async function processLifecycle(record: CaptureRecord, file: CaptureFile, actor: Actor, state: RunState): Promise<void> {
  const parsed = lifecycleSchema.safeParse(record);
  if (!parsed.success) {
    state.report.unsupported++;
    markIncomplete(state);
    return;
  }
  const event = parsed.data.event as unknown as TracingEvent;
  if (event.type === "turn_start") {
    if (parsed.data.event.turnIndex === 0) {
      await endTurn(actor, record, file, state, "turn_boundary_reconstructed_from_turn_start_zero");
      actor.turnPending = true;
    }
    recordImported(state);
    return;
  }
  if (event.type === "session_shutdown") {
    await endTurn(actor, record, file, state, "session_shutdown_without_agent_end_archive");
    await emit(actor, event, captureSeed(actor, record, "session-shutdown"), currentProvenance(
      file, ["session_shutdown"], [], ["original_agent_end_event"],
    ));
    actor.turnPending = false;
    for (const request of actor.pendingRequests.values()) if (!request.completed) markIncomplete(state, request.captureId);
    for (const tool of actor.pendingTools.values()) markIncomplete(state, tool.requestId);
    recordImported(state);
    actor.pendingRequests.clear();
    actor.pendingTools.clear();
    actor.history = [];
    actor.entries = [];
    actor.hasHistory = false;
    return;
  }
  if (SUPPORTED_LIFECYCLE.has(event.type)) {
    await emit(actor, event, captureSeed(actor, record, event.type), currentProvenance(
      file, [event.type], [], ["unarchived_lifecycle_context"],
    ));
    recordImported(state);
  }
}

async function processCompaction(record: CaptureRecord, file: CaptureFile, actor: Actor, state: RunState): Promise<void> {
  const parsed = compactionSchema.safeParse(record);
  if (!parsed.success) {
    state.report.unsupported++;
    markIncomplete(state);
    return;
  }
  await emit(actor, {
    type: "session_compact",
    compactionEntry: parsed.data.entry,
    fromExtension: parsed.data.fromExtension,
  } as unknown as TracingEvent,
  captureSeed(actor, record, "session-compact"), currentProvenance(
    file,
    ["compaction_entry", "from_extension"],
    [],
    ["compaction_start_timestamp", "summarizer_usage"],
  ));
  recordImported(state);
}

async function processRecord(
  raw: unknown,
  file: CaptureFile,
  state: RunState,
  actors: Map<string, Actor>,
): Promise<{ requestId?: string; requestSeen: boolean }> {
  const record = parseEnvelope(raw);
  if (!record) {
    state.report.skipped++;
    state.incompleteRecords++;
    return { requestSeen: file.kind === "request" };
  }
  state.report.processed++;
  if (record.schemaVersion !== 2 || record.kind !== file.kind || record.eventSequence !== file.sequence) {
    state.report.unsupported++;
    state.incompleteRecords++;
    return { requestSeen: record.kind === "request" || file.kind === "request" };
  }
  if (!validateRecordDetails(record)) {
    state.report.skipped++;
    state.incompleteRecords++;
    return { requestSeen: record.kind === "request" };
  }
  const sessionId = stringField(record.sessionId);
  if (!sessionId) {
    state.report.unsupported++;
    markIncomplete(state);
    return { requestSeen: record.kind === "request" };
  }
  if (state.sessions && !state.sessions.has(sessionId)) {
    state.report.skipped++;
    return { requestSeen: false };
  }
  const key = `${record.runtime.id}\u0000${sessionId}`;
  let actor = actors.get(key);
  if (!actor) {
    actor = createActor(record, file, state);
    if (!actor) {
      state.report.skipped++;
      state.incompleteRecords++;
      return { requestSeen: record.kind === "request" };
    }
    actors.set(key, actor);
  }
  updateActorIdentity(actor, record, file);

  if (record.kind === "request") {
    state.report.requests++;
    const result = await processRequest(record, file, actor, state);
    return { requestId: result.requestId, requestSeen: true };
  }
  if (record.kind === "response") await processResponse(record, file, actor, state);
  else if (record.kind === "message") await processMessage(record, file, actor, state);
  else if (record.kind === "lifecycle") await processLifecycle(record, file, actor, state);
  else if (record.kind === "compaction") await processCompaction(record, file, actor, state);
  else if (record.kind === "gap") {
    markIncomplete(state);
    state.report.warnings++;
  } else if (record.kind === "snapshot") {
    // Snapshots contain prompt hashes and metadata, not reconstructable prompt text.
  } else if (record.kind === "wire_request") {
    // This is a transport mirror of a request record, never a second generation.
    state.report.skipped++;
  } else if (record.kind === "retention" || record.kind === "summary") {
    state.report.unsupported++;
    state.report.skipped++;
  } else {
    state.report.unsupported++;
    state.incompleteRecords++;
  }
  return { requestSeen: false };
}

function finishIncompleteCounts(state: RunState, actors: Map<string, Actor>): void {
  for (const actor of actors.values()) {
    for (const request of actor.pendingRequests.values()) {
      if (!request.completed) markIncomplete(state, request.captureId);
    }
    for (const tool of actor.pendingTools.values()) markIncomplete(state, tool.requestId);
    for (const root of actor.roots) {
      if (root.sourceRequests.some(captureId => state.incompleteCaptures.has(captureId))) root.incomplete = true;
    }
  }
  state.report.incomplete = state.incompleteCaptures.size + state.incompleteRecords;
}

function updateReportExpectations(state: RunState, actors: Map<string, Actor>): void {
  const { report } = state;
  const roots = [...actors.values()].flatMap(actor => actor.roots);
  report.sourceSpanExpectations = roots.map(root => ({ ...root, sourceRequests: [...root.sourceRequests] }));
  report.rootIds = [...(report.dryRun ? new Set(roots.map(root => root.traceId)) : state.traceIds)];
  report.observationIds = [...state.observationIds];
}

function makeReport(root: string, dryRun: boolean, policyDisabled: boolean): BackfillReport {
  return {
    root,
    dryRun,
    policyDisabled,
    processed: 0,
    imported: 0,
    wouldImport: 0,
    skipped: 0,
    unsupported: 0,
    unmatched: 0,
    incomplete: 0,
    requests: 0,
    warnings: 0,
    rootIds: [],
    observationIds: [],
    sourceSpanExpectations: [],
  };
}

function validateOptions(options: BackfillOptions): { root: string; from?: number; until?: number; limit?: number } {
  const root = path.resolve(options.root ?? DEFAULT_ROOT());
  const from = isoTime(options.from, "from");
  const until = isoTime(options.until, "until");
  if (from !== undefined && until !== undefined && from > until) throw new Error("--from must not be later than --until");
  if (options.limit !== undefined && (!Number.isSafeInteger(options.limit) || options.limit < 1)) {
    throw new Error("--limit must be a positive integer");
  }
  const sessions = options.sessions?.map(value => value.trim()).filter(Boolean);
  if (options.observationIds && (options.observationIds.length === 0 ||
    options.observationIds.some(id => !/^[0-9a-f]{16}$/.test(id)))) {
    throw new Error("Observation recovery requires non-empty, exact 16-hex SDK span IDs");
  }
  if (options.sessions && (!sessions || sessions.length === 0)) throw new Error("--session requires a non-empty session ID");
  return { root, from, until, limit: options.limit };
}

/** Validate and replay retained schema-2 raw captures through the normal tracing consumer. */
export async function runBackfill(options: BackfillOptions = {}): Promise<BackfillReport> {
  const startupTime = Date.now();
  const validated = validateOptions(options);
  const dryRun = options.dryRun === true;
  const enabled = tracingEnabled();
  if (!dryRun && !enabled) throw new Error("Backfill export is disabled by LANGFUSE_TRACING_ENABLED=false");
  const config = dryRun || !enabled ? undefined : loadConfig();
  if (!dryRun && enabled && !config) {
    throw new Error("Tracing is enabled but Langfuse credentials/configuration are unavailable");
  }
  const report = makeReport(validated.root, dryRun, !enabled);
  const until = validated.until ?? startupTime;
  let files = await enumerateCaptureFiles(validated.root, startupTime, validated.from, until, report);
  const state: RunState = {
    report,
    observationIds: new Set(),
    traceIds: new Set(),
    includeObservationIds: options.observationIds ? new Set(options.observationIds) : undefined,
    incompleteCaptures: new Set(),
    incompleteRecords: report.incomplete,
    sessions: options.sessions ? new Set(options.sessions.map(value => value.trim()).filter(Boolean)) : undefined,
    config,
    canExport: !dryRun && enabled && !!config,
    dryRun,
    stopAfterCurrentMessage: false,
  };
  const actors = new Map<string, Actor>();
  let selectedRequests = 0;
  let requestAfterLimit: string | undefined;
  let snapshot: CaptureSnapshot | undefined;

  try {
    if (!dryRun) {
      snapshot = await snapshotCaptures(files,
        () => { report.skipped++; state.incompleteRecords++; },
        { sessions: state.sessions, limit: validated.limit, onSkip: () => { report.skipped++; } });
      files = snapshot.files;
      console.error(`Input snapshot: ${files.length} files at ${snapshot.root}`);
    }
    for (let index = 0; index < files.length; index++) {
      const file = files[index]!;
      if (state.sessions) {
        let peeked: string | undefined;
        try {
          peeked = await peekSessionId(file);
        } catch (error) {
          report.skipped++;
          markIncomplete(state);
          if ((error as NodeJS.ErrnoException).code !== "ENOENT") report.warnings++;
          continue;
        }
        if (peeked && !state.sessions.has(peeked)) {
          report.skipped++;
          continue;
        }
      }
      if (validated.limit !== undefined && selectedRequests >= validated.limit && file.kind === "request") {
        report.skipped += files.length - index;
        break;
      }
      if (state.stopAfterCurrentMessage) {
        report.skipped += files.length - index;
        break;
      }
      if (requestAfterLimit && file.kind === "request") {
        // A bounded smoke pass ends before the next provider request; tool results in that later
        // request are intentionally incomplete rather than exceeding the request limit.
        report.skipped += files.length - index;
        break;
      }
      let raw: unknown;
      try {
        raw = await readCapture(file);
      } catch (error) {
        report.skipped++;
        state.incompleteRecords++;
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") report.warnings++;
        continue;
      }
      const envelope = parseEnvelope(raw);
      if (envelope?.kind === "request" && (!state.sessions || !envelope.sessionId || state.sessions.has(envelope.sessionId))) {
        selectedRequests++;
      }
      const processed = await processRecord(raw, file, state, actors);
      // forceFlush exports its queued batches concurrently. Keep the producer
      // window small enough to respect the SDK exporter's concurrency ceiling.
      if (state.canExport && (index + 1) % 10 === 0) await flushTracing();
      if ((index + 1) % 1000 === 0) {
        console.error(`Backfill: ${index + 1}/${files.length} files, ${report.requests} requests, ${report.imported} replayed records`);
      }
      if (validated.limit !== undefined && selectedRequests === validated.limit && processed.requestId) {
        requestAfterLimit = processed.requestId;
        state.currentLimitRequestId = processed.requestId;
      }
      if (requestAfterLimit && file.kind === "message") {
        const record = envelope;
        const correlation = record ? parseCorrelation(record) : undefined;
        if (correlation?.captureId === requestAfterLimit) state.stopAfterCurrentMessage = true;
      }
    }

    for (const actor of actors.values()) {
      if (actor.turnOpen) {
        const syntheticRecord: CaptureRecord = {
          schemaVersion: 2,
          kind: "lifecycle",
          ts: actor.currentTime.toISOString(),
          eventSequence: 1,
          runtime: { id: actor.runtimeId },
          sessionId: actor.sessionId,
          cwd: actor.cwd,
          agent: actor.agent,
          event: { type: "session_shutdown" },
        };
        await endTurn(actor, syntheticRecord, actor.currentFile, state, "end_of_archive_without_agent_end_callback");
      }
      for (const request of actor.pendingRequests.values()) if (!request.completed) markIncomplete(state, request.captureId);
      for (const tool of actor.pendingTools.values()) markIncomplete(state, tool.requestId);
      if (state.canExport) await flushTracing();
    }
    finishIncompleteCounts(state, actors);
    updateReportExpectations(state, actors);
    if (state.canExport) {
      await flushTracing();
      await shutdownTracing();
    }
    if (snapshot) {
      if (options.keepSnapshot) report.snapshotRoot = snapshot.root;
      else await fs.rm(snapshot.root, { recursive: true, force: true });
    }
    return report;
  } catch (error) {
    console.error(`Backfill stopped after ${report.processed} files and ${report.requests} requests; SDK delivery is not fully confirmed.`);
    if (snapshot) console.error(`Input snapshot retained for retry: ${snapshot.root}`);
    if (state.canExport) {
      try {
        await shutdownTracing();
      } catch {
        // Preserve the original replay/export error; shutdown errors are reported if shutdown was primary.
      }
    }
    throw error;
  }
}

const HELP = `Usage: bun run backfill [options]

Replay retained schema-2 raw request captures through the existing Langfuse tracing consumer.

Options:
  --root <path>       Raw capture root (default: dirname(getAgentDir())/logs/raw-requests)
  --session <id>      Include this session; repeatable
  --from <ISO time>   Include captures at or after this timestamp
  --until <ISO time>  Include captures at or before this timestamp (default: startup cutoff)
  --limit <number>    Replay at most this many provider request records
  --dry-run           Validate and report without SDK/exporter construction or credentials
  --keep-snapshot     Keep the private input snapshot after export for backend verification
  --only-observations <file> JSON array of exact SDK observation IDs for targeted recovery
  --help              Show this help
`;

async function parseArgs(args: string[]): Promise<BackfillOptions & { help?: boolean }> {
  const options: BackfillOptions & { help?: boolean } = {};
  for (let index = 0; index < args.length; index++) {
    const flag = args[index]!;
    if (flag === "--help" || flag === "-h") {
      options.help = true;
      continue;
    }
    if (flag === "--dry-run") {
      options.dryRun = true;
      continue;
    }
    if (flag === "--keep-snapshot") {
      options.keepSnapshot = true;
      continue;
    }
    const value = args[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`${flag} requires a value`);
    index++;
    if (flag === "--root") options.root = value;
    else if (flag === "--session") options.sessions = [...(options.sessions ?? []), value];
    else if (flag === "--from") options.from = value;
    else if (flag === "--until") options.until = value;
    else if (flag === "--only-observations") {
      const ids: unknown = JSON.parse(await fs.readFile(value, "utf8"));
      if (!Array.isArray(ids) || ids.some(id => typeof id !== "string")) throw new Error("Observation ID file must contain a JSON string array");
      options.observationIds = ids;
    }
    else if (flag === "--limit") {
      const limit = Number(value);
      if (!Number.isSafeInteger(limit)) throw new Error("--limit must be a positive integer");
      options.limit = limit;
    } else throw new Error(`Unknown option ${flag}`);
  }
  return options;
}

async function main(): Promise<void> {
  try {
    const options = await parseArgs(process.argv.slice(2));
    if (options.help) {
      console.log(HELP);
      return;
    }
    const report = await runBackfill(options);
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    const message = error instanceof Error ? error.message : "unknown failure";
    console.error(`Raw capture backfill failed: ${message}`);
    process.exitCode = 1;
  }
}

if (Reflect.get(import.meta, "main") === true) void main();
