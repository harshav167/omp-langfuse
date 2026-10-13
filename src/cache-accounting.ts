import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { analyzePayloadOnce, historyContains, object, type RequestShape } from "./raw-capture-core.ts";

/**
 * Per-request prompt-cache accounting and model/account switch tracking, shared by
 * raw capture and Langfuse tracing so both report the same numbers.
 *
 * A session-wide "cached share" hides losses: one full re-read of a 600k-token
 * prefix disappears inside hundreds of cached turns. Each successful reply is
 * therefore compared with the previous successful reply of the same model in the
 * same session: what that request left reusable, what this one read from cache,
 * and why any difference occurred.
 */

/** Gap treated as cache expiry: the shortest common provider retention (5 minutes). */
export const IDLE_EXPIRY_MS = 5 * 60 * 1000;
/** Reusable tokens a request may miss before it counts as a loss; absorbs the trailing token and block rounding. */
export const LOSS_THRESHOLD_TOKENS = 1024;
const MAX_SESSIONS = 64;

export type CacheClass =
 | "first-observed"
 | "append-only-progression"
 | "prefix-rewrite-no-loss"
 | "model-switch"
 | "account-switch"
 | "compaction"
 | "prefix-rewrite"
 | "ttl-sized-idle"
 | "after-interrupt"
 | "stalled-tail"
 | "unexplained-loss";

export interface ModelIdentity {
 provider: string;
 model: string;
}
export interface ModelSwitch {
 from: ModelIdentity;
 to: ModelIdentity;
}
export interface AccountSwitch {
 provider: string;
 from: number;
 to: number;
}
export interface SessionCacheTotals {
 requests: number;
 losses: number;
 missedReusable: number;
 cacheRead: number;
 promptTokens: number;
}

export interface CacheAssessment {
 provider: string;
 model: string;
 credentialId?: number;
 promptTokens: number;
 cacheRead: number;
 cacheWrite: number;
 uncached: number;
 /** No earlier successful reply of this model in this session: the whole prompt is necessarily uncached. */
 coldStart: boolean;
 /** Prompt of the previous successful reply of this model: everything that could have been read from cache. */
 reusable?: number;
 /** Reusable tokens this request did not read from cache (0 below {@link LOSS_THRESHOLD_TOKENS}). */
 missedReusable: number;
 /** Prompt tokens beyond the reusable prefix: content that is new since the previous reply. */
 newTokens: number;
 basis: "first-observed" | "implicit-prefix" | "explicit-breakpoints";
 classification: CacheClass;
 loss: boolean;
 /** The previous request's history is unchanged at the start of this one; undefined when either was not observed. */
 appendOnly?: boolean;
 /** Time from the previous successful reply of this model to this request's dispatch. */
 idleMs?: number;
 /** An attempt failed, was aborted or was re-sent since the previous successful reply. */
 disrupted: boolean;
 accountSwitch?: AccountSwitch;
 modelSwitch?: ModelSwitch;
 session: SessionCacheTotals;
}

interface Attempt {
 shape?: RequestShape;
 startedAt: number;
}
interface Success {
 promptTokens: number;
 cacheRead: number;
 shape?: RequestShape;
 endedAt: number;
}
interface SessionState {
 pending: Attempt[];
 lastByModel: Map<string, Success>;
 lastSuccessModel?: ModelIdentity;
 lastRequestModel?: ModelIdentity;
 credentials: Map<string, number>;
 disrupted: boolean;
 compacted: boolean;
 totals: SessionCacheTotals;
}

interface AssistantFacts {
 provider: string;
 model: string;
 credentialId?: number;
 stopReason?: string;
 duration?: number;
 usage?: { input: number; cacheRead: number; cacheWrite: number };
}

const sessions = new Map<string, SessionState>();
const assessments = new WeakMap<object, CacheAssessment>();
const modelSwitches = new WeakMap<object, ModelSwitch>();

/** Assessment of a reply, available to every `message_end` handler registered after this module. */
export function cacheAssessmentFor(message: unknown): CacheAssessment | undefined {
 const key = object(message);
 return key ? assessments.get(key) : undefined;
}

/** Model change observed at dispatch of this request payload, relative to the previous request of the session. */
export function modelSwitchFor(payload: unknown): ModelSwitch | undefined {
 const key = object(payload);
 return key ? modelSwitches.get(key) : undefined;
}

/** OMP's generated (or user-set) session title, when the host exposes it. */
export function sessionTitleOf(ctx: { sessionManager?: { getSessionName?: () => string | undefined } }): string | undefined {
 try {
  const title = ctx.sessionManager?.getSessionName?.();
  return typeof title === "string" && title.trim() ? title.trim() : undefined;
 } catch {
  return undefined;
 }
}

/** `provider/model`: the identity used for per-model cache state, switch events and metadata. */
export function modelLabel(model: ModelIdentity): string {
 return `${model.provider}/${model.model}`;
}

/** Flat Langfuse metadata for an assessment. */
export function cacheMetadata(cache: CacheAssessment): Record<string, unknown> {
 return {
  cache_class: cache.classification,
  cache_basis: cache.basis,
  cache_cold_start: cache.coldStart,
  ...(cache.reusable !== undefined ? { cache_reusable_tokens: cache.reusable } : {}),
  cache_missed_reusable_tokens: cache.missedReusable,
  cache_new_tokens: cache.newTokens,
  ...(cache.idleMs !== undefined ? { cache_idle_ms: cache.idleMs } : {}),
  ...(cache.appendOnly !== undefined ? { cache_append_only: cache.appendOnly } : {}),
  ...(cache.disrupted ? { cache_after_disruption: true } : {}),
  ...(cache.accountSwitch
   ? { account_switch_from: cache.accountSwitch.from, account_switch_to: cache.accountSwitch.to }
   : {}),
  ...(cache.modelSwitch
   ? { model_switch_from: modelLabel(cache.modelSwitch.from), model_switch_to: modelLabel(cache.modelSwitch.to) }
   : {}),
  session_cache_requests: cache.session.requests,
  session_cache_losses: cache.session.losses,
  session_missed_reusable_tokens: cache.session.missedReusable,
 };
}

function sessionState(sessionId: string): SessionState {
 let state = sessions.get(sessionId);
 if (state) {
  sessions.delete(sessionId);
  sessions.set(sessionId, state);
  return state;
 }
 state = {
  pending: [],
  lastByModel: new Map(),
  credentials: new Map(),
  disrupted: false,
  compacted: false,
  totals: { requests: 0, losses: 0, missedReusable: 0, cacheRead: 0, promptTokens: 0 },
 };
 sessions.set(sessionId, state);
 if (sessions.size > MAX_SESSIONS) {
  const oldest = sessions.keys().next().value;
  if (oldest !== undefined) sessions.delete(oldest);
 }
 return state;
}

function assistantFacts(message: unknown): AssistantFacts | undefined {
 const m = object(message);
 if (!m || m.role !== "assistant" || typeof m.provider !== "string" || typeof m.model !== "string") return undefined;
 const usage = object(m.usage);
 const count = (value: unknown) => (typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0);
 return {
  provider: m.provider,
  model: m.model,
  credentialId: typeof m.credentialId === "number" ? m.credentialId : undefined,
  stopReason: typeof m.stopReason === "string" ? m.stopReason : undefined,
  duration: typeof m.duration === "number" ? m.duration : undefined,
  usage: usage ? { input: count(usage.input), cacheRead: count(usage.cacheRead), cacheWrite: count(usage.cacheWrite) } : undefined,
 };
}

function classifyLoss(
 state: SessionState,
 previous: Success,
 cacheRead: number,
 signals: { modelSwitch: boolean; accountSwitch: boolean; appendOnly?: boolean; idleMs: number },
): CacheClass {
 if (signals.modelSwitch) return "model-switch";
 if (signals.accountSwitch) return "account-switch";
 if (state.compacted) return "compaction";
 if (signals.appendOnly === false) return "prefix-rewrite";
 if (signals.idleMs >= IDLE_EXPIRY_MS) return "ttl-sized-idle";
 if (state.disrupted) return "after-interrupt";
 // The cache still holds what it held before but did not advance: the uncached tail is re-processed.
 if (cacheRead > 0 && cacheRead >= previous.cacheRead) return "stalled-tail";
 return "unexplained-loss";
}

/**
 * Interrupted or failed replies are not assessed: their usage is partial or reported
 * before the cache split. They mark the next assessment as disrupted instead.
 */
function assess(state: SessionState, attempt: Attempt | undefined, facts: AssistantFacts, at: number): CacheAssessment | undefined {
 if (facts.stopReason === "aborted" || facts.stopReason === "error" || !facts.usage) return undefined;
 const { input: uncached, cacheRead, cacheWrite } = facts.usage;
 const promptTokens = uncached + cacheRead + cacheWrite;
 if (promptTokens <= 0) return undefined;
 const model: ModelIdentity = { provider: facts.provider, model: facts.model };
 const modelKey = modelLabel(model);
 const previous = state.lastByModel.get(modelKey);
 const last = state.lastSuccessModel;
 const modelSwitch =
  last && (last.provider !== model.provider || last.model !== model.model) ? { from: last, to: model } : undefined;
 const priorCredential = state.credentials.get(model.provider);
 const accountSwitch =
  priorCredential !== undefined && facts.credentialId !== undefined && priorCredential !== facts.credentialId
   ? { provider: model.provider, from: priorCredential, to: facts.credentialId }
   : undefined;
 if (facts.credentialId !== undefined) state.credentials.set(model.provider, facts.credentialId);
 const explicit = cacheWrite > 0 || (attempt?.shape?.markers.length ?? 0) > 0;

 let classification: CacheClass;
 let reusable: number | undefined;
 let missedReusable = 0;
 let newTokens = promptTokens;
 let appendOnly: boolean | undefined;
 let idleMs: number | undefined;
 if (!previous) {
  classification = modelSwitch ? "model-switch" : "first-observed";
 } else {
  reusable = previous.promptTokens;
  const missed = Math.max(0, reusable - cacheRead);
  missedReusable = missed > LOSS_THRESHOLD_TOKENS ? missed : 0;
  newTokens = Math.max(0, promptTokens - reusable);
  if (attempt?.shape && previous.shape) {
   appendOnly =
    historyContains(previous.shape.items, attempt.shape.items) &&
    previous.shape.toolsHash === attempt.shape.toolsHash &&
    previous.shape.systemHash === attempt.shape.systemHash;
  }
  const dispatchedAt = attempt?.startedAt ?? at - (facts.duration ?? 0);
  idleMs = Math.max(0, dispatchedAt - previous.endedAt);
  classification = missedReusable
   ? classifyLoss(state, previous, cacheRead, { modelSwitch: !!modelSwitch, accountSwitch: !!accountSwitch, appendOnly, idleMs })
   : appendOnly === false
    ? "prefix-rewrite-no-loss"
    : "append-only-progression";
 }
 const loss = missedReusable > 0;
 const disrupted = state.disrupted;
 const totals = state.totals;
 totals.requests++;
 totals.cacheRead += cacheRead;
 totals.promptTokens += promptTokens;
 if (loss) {
  totals.losses++;
  totals.missedReusable += missedReusable;
 }
 state.lastByModel.set(modelKey, { promptTokens, cacheRead, shape: attempt?.shape, endedAt: at });
 state.lastSuccessModel = model;
 state.disrupted = false;
 state.compacted = false;
 return {
  provider: model.provider,
  model: model.model,
  ...(facts.credentialId !== undefined ? { credentialId: facts.credentialId } : {}),
  promptTokens,
  cacheRead,
  cacheWrite,
  uncached,
  coldStart: !previous,
  ...(reusable !== undefined ? { reusable } : {}),
  missedReusable,
  newTokens,
  basis: !previous ? "first-observed" : explicit ? "explicit-breakpoints" : "implicit-prefix",
  classification,
  loss,
  ...(appendOnly !== undefined ? { appendOnly } : {}),
  ...(idleMs !== undefined ? { idleMs } : {}),
  disrupted,
  ...(accountSwitch ? { accountSwitch } : {}),
  ...(modelSwitch ? { modelSwitch } : {}),
  session: { ...totals },
 };
}

function guard(handler: () => void): void {
 try {
  handler();
 } catch {
  // Accounting is observational; it must never disturb the host's request path.
 }
}

/**
 * Registers before raw capture and tracing so their handlers for the same event can
 * read the assessment and switch records produced here.
 */
export function registerCacheAccounting(pi: Pick<ExtensionAPI, "on">, options: { now?: () => number } = {}): void {
 const now = options.now ?? Date.now;
 pi.on("before_provider_request", (event, ctx) =>
  guard(() => {
   // Codex prewarm requests are never answered by an assistant message.
   if (object(event)?.isPrewarm === true) return;
   const state = sessionState(ctx.sessionManager.getSessionId());
   let shape: RequestShape | undefined;
   try {
    shape = analyzePayloadOnce(event.payload);
   } catch {
    shape = undefined;
   }
   // OMP sends one request at a time per session: anything still unanswered failed or was re-sent.
   if (state.pending.length) state.disrupted = true;
   state.pending = [{ shape, startedAt: now() }];
   const model = ctx.model ? { provider: ctx.model.provider, model: ctx.model.id } : undefined;
   const payload = object(event.payload);
   const previousModel = state.lastRequestModel;
   if (model && payload && previousModel && (previousModel.provider !== model.provider || previousModel.model !== model.model)) {
    modelSwitches.set(payload, { from: previousModel, to: model });
   }
   if (model) state.lastRequestModel = model;
  }),
 );
 pi.on("message_end", (event, ctx) =>
  guard(() => {
   const facts = assistantFacts(event.message);
   if (!facts) return;
   const state = sessionState(ctx.sessionManager.getSessionId());
   const attempt = state.pending.at(-1);
   state.pending = [];
   const assessment = assess(state, attempt, facts, now());
   if (!assessment) {
    state.disrupted = true;
    return;
   }
   const key = object(event.message);
   if (key) assessments.set(key, assessment);
  }),
 );
 pi.on("session_compact", (_event, ctx) =>
  guard(() => {
   sessionState(ctx.sessionManager.getSessionId()).compacted = true;
  }),
 );
 pi.on("session_shutdown", (_event, ctx) =>
  guard(() => {
   sessions.delete(ctx.sessionManager.getSessionId());
  }),
 );
}
