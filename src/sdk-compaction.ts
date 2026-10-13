import type { AgentSession, ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import { lookup } from "@oh-my-pi/pi-coding-agent/config/registry";
import type { Span, SpanContext, Tracer } from "@opentelemetry/api";
import type { PiUsage } from "./index.ts";

// Resolve callback types through the declared host dependency's public session API.
type AgentTelemetryConfig = NonNullable<AgentSession["agent"]["telemetry"]>;
type ChatUsageEvent = Parameters<NonNullable<AgentTelemetryConfig["onChatUsage"]>>[0];

export interface SdkCompactionUsage {
  kind: string;
  startedAt: Date;
  endedAt: Date;
  model: string;
  provider?: string;
  usage?: PiUsage;
  costDetails?: Record<string, number>;
}

function attributesOf(span: Span): Record<string, unknown> | undefined {
  if ("attributes" in span && span.attributes && typeof span.attributes === "object") {
    return span.attributes as Record<string, unknown>;
  }
  return undefined;
}

/** Listen to the SDK's real oneshot accounting without changing global OTLP or content capture. */
export function registerSdkCompactionUsage(
  pi: ExtensionAPI,
  tracer: () => Tracer,
  parentAtStart: () => SpanContext | undefined,
  onUsage: (event: SdkCompactionUsage, ctx: ExtensionContext, parent: SpanContext | undefined) => void,
): void {
  let restore: (() => void) | undefined;
  const diagnose = (message: string, error?: unknown): void => {
    if (process.env.PI_LANGFUSE_DEBUG !== "true") return;
    try {
      pi.logger.warn(`langfuse: ${message}`, error === undefined ? {} : { error: String(error) });
    } catch {
      // Diagnostics are optional too; never replace an SDK callback's failure.
    }
  };
  const detach = (): void => {
    const previous = restore;
    restore = undefined;
    try { previous?.(); }
    catch (error) { diagnose("native compaction hook could not be restored", error); }
  };

  pi.on("session_start", (_event, ctx) => {
    detach();
    try {
      const session = pi.pi.AgentRegistry.global().list()
        .find(ref => ref.session?.sessionManager.getSessionId() === ctx.sessionManager.getSessionId())?.session;
      if (!session) return;
      const exportGate = lookup("telemetry.otlpExportEnabled");
      if (exportGate?.get(session.settings) === false ||
        process.env.OTEL_SDK_DISABLED?.trim().toLowerCase() === "true" ||
        process.env.OTEL_TRACES_EXPORTER?.split(",").some(value => value.trim().toLowerCase() === "none")) return;

      const agent = session.agent;
      const previous = agent.telemetry;
      const starts = new Map<string, { startedAt: Date; parent?: SpanContext; model?: string; provider?: string }>();
      const usages = new Map<string, ChatUsageEvent>();
      let active = true;
      const capture = (operation: () => void): void => {
        if (!active) return;
        try { operation(); }
        catch (error) { diagnose("native compaction accounting unavailable", error); }
      };
      const configured: AgentTelemetryConfig = {
        ...previous,
        tracer: previous?.tracer ?? tracer(),
        // Preserve the SDK's existing content policy, including its environment default.
        costEstimator: previous?.costEstimator ?? (event => event.usageCost.total > 0
          ? { usd: event.usageCost.total }
          : undefined),
        onSpanStart(event) {
          try { return previous?.onSpanStart?.(event); }
          finally {
            capture(() => {
              if (event.kind === "chat" && event.stepNumber === -1) {
                starts.set(event.span.spanContext().spanId, {
                  startedAt: new Date(), parent: parentAtStart(), model: event.model?.id, provider: event.model?.provider,
                });
              }
            });
          }
        },
        onChatUsage(event) {
          capture(() => {
            const id = event.span.spanContext().spanId;
            if (starts.has(id)) usages.set(id, event);
          });
          return previous?.onChatUsage?.(event);
        },
        onSpanEnd(event) {
          try { return previous?.onSpanEnd?.(event); }
          finally {
            capture(() => {
              const id = event.span.spanContext().spanId;
              const start = starts.get(id);
              const usageEvent = usages.get(id);
              starts.delete(id);
              usages.delete(id);
              if (!start) return;
              const attributes = attributesOf(event.span);
              if (!attributes) { diagnose("native compaction span attributes unavailable"); return; }
              const kind = attributes["omp.gen_ai.oneshot.kind"];
              if (typeof kind !== "string" || !kind.startsWith("compaction_")) return;
              const usage = usageEvent?.usage;
              const cacheRead = usage?.cachedInputTokens ?? 0;
              const cacheWrite = usage?.cacheWriteTokens ?? 0;
              const cost = usageEvent?.cost;
              onUsage({
                kind, startedAt: start.startedAt, endedAt: new Date(),
                model: usageEvent?.model ?? start.model ?? "",
                provider: usageEvent?.provider ?? start.provider,
                ...(usage ? { usage: {
                  input: usage.inputTokens - cacheRead - cacheWrite,
                  output: usage.outputTokens, cacheRead, cacheWrite,
                  ...(usage.reasoningOutputTokens === undefined ? {} : { reasoningTokens: usage.reasoningOutputTokens }),
                } } : {}),
                ...(cost && "usd" in cost ? { costDetails: { total: cost.usd } } : {}),
              }, ctx, start.parent);
            });
          }
        },
      };
      restore = () => {
        active = false;
        starts.clear();
        usages.clear();
        if (agent.telemetry === configured) agent.setTelemetry(previous);
      };
      agent.setTelemetry(configured);
    } catch (error) {
      detach();
      diagnose("native compaction integration unavailable", error);
    }
  });
  pi.on("session_shutdown", detach);
}
