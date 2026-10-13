import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { TracingAPI, TracingContext, TracingEvent as ExtensionEvent } from "../src/index.ts";
import { flushTracing, registerTracing, shutdownTracing } from "../src/index.ts";
import type { CapturedSpan } from "./helpers.ts";
import { startCaptureServer } from "./helpers.ts";

async function fixture() {
  const capture = await startCaptureServer();
  const savedEnabled = process.env.LANGFUSE_TRACING_ENABLED;
  process.env.LANGFUSE_TRACING_ENABLED = "true";
  const entries: Array<{ type: string; message: { role: string; content: string } }> = [];
  const ctx: TracingContext = {
    agent: { kind: "main", id: "OrderingMain", name: "main", depth: 0 },
    sessionManager: { getSessionId: () => "ordering-session", getEntries: () => entries, getSessionName: () => "Ordering fixture" },
    cwd: "/fixture", hasUI: false, getSystemPrompt: () => [],
  };
  const handlers = new Map<ExtensionEvent["type"], Array<(event: ExtensionEvent) => void | Promise<void>>>();
  const api: TracingAPI = {
    on<E extends ExtensionEvent["type"]>(type: E, handler: (event: Extract<ExtensionEvent, { type: E }>, ctx: TracingContext) => void | Promise<void>) {
      const dispatch = (event: ExtensionEvent) => {
        if (event.type !== type) return;
        // The checked discriminator correlates the union with this registration's E.
        const matching = event as Extract<ExtensionEvent, { type: E }>;
        return handler(matching, ctx);
      };
      handlers.set(type, [...(handlers.get(type) ?? []), dispatch]);
    },
    getThinkingLevel: () => undefined,
    getAllTools: () => [],
    getActiveTools: () => [],
  };
  const emit = async (event: ExtensionEvent) => {
    for (const handler of handlers.get(event.type) ?? []) await handler(event);
  };
  const prompt = async (text: string) => {
    entries.push({ type: "message", message: { role: "user", content: text } });
    await emit({ type: "before_agent_start", prompt: text, systemPrompt: [] });
  };
  const tool = async () => {
    await emit({ type: "tool_execution_start", toolCallId: "call-1", toolName: "bash", args: { command: "fixture" } });
    await emit({ type: "tool_execution_end", toolCallId: "call-1", toolName: "bash", result: { content: [{ type: "text", text: "fixture result" }] }, isError: false });
  };
  return {
    api, capture, emit, prompt, tool,
    register: () => registerTracing(api, { publicKey: "pk-lf-test", secretKey: "sk-lf-test", baseUrl: `http://127.0.0.1:${capture.port}` }),
    close: async () => {
      try {
        await emit({ type: "session_shutdown" });
        await shutdownTracing();
      } finally {
        capture.close();
        if (savedEnabled === undefined) delete process.env.LANGFUSE_TRACING_ENABLED;
        else process.env.LANGFUSE_TRACING_ENABLED = savedEnabled;
      }
    },
  };
}

function rootFor(spans: CapturedSpan[], prompt: string): CapturedSpan | undefined {
  return spans.find(span => {
    if (span.name !== "omp-agent") return false;
    const raw = span.attrs["langfuse.observation.input"];
    if (typeof raw !== "string") return false;
    const input: unknown = JSON.parse(raw);
    return !!input && typeof input === "object" && "content" in input && input.content === prompt;
  });
}

function assertToolParent(spans: CapturedSpan[], prompt: string) {
  const root = rootFor(spans, prompt);
  const tool = spans.find(span => span.name === "bash");
  assert.ok(root);
  assert.ok(tool, "the interrupted turn must not discard the new prompt's tool");
  assert.equal(tool.parentSpanId, root.spanId);
  assert.equal(tool.attrs["langfuse.observation.output"], "fixture result");
  assert.ok(root.endNs >= tool.endNs, "the root must cover its completed tool");
}

describe("agent_end ordering", () => {
  for (const lateEndAfterStart of [false, true]) {
    it(`keeps the replacement trace when the old end arrives ${lateEndAfterStart ? "after" : "before"} its agent_start`, async () => {
      const f = await fixture();
      try {
        f.register();
        await f.prompt("first");
        await f.emit({ type: "agent_start" });
        await f.prompt("replacement");
        if (lateEndAfterStart) await f.emit({ type: "agent_start" });
        await f.emit({ type: "agent_end", messages: [] });
        if (!lateEndAfterStart) await f.emit({ type: "agent_start" });
        await f.tool();
        await f.emit({ type: "agent_end", messages: [] });
        await flushTracing();
        const spans = f.capture.spans();
        assert.equal(spans.filter(span => span.name === "omp-agent").length, 2);
        assertToolParent(spans, "replacement");
        await f.tool();
        await flushTracing();
        assert.equal(f.capture.spans().filter(span => span.name === "bash").length, 1, "the settled trace must not keep accepting tools");
      } finally {
        await f.close();
      }
    });
  }

  it("drains multiple superseded runs without closing the latest prompt", async () => {
    const f = await fixture();
    try {
      f.register();
      for (const prompt of ["first", "second", "latest"]) {
        await f.prompt(prompt);
        await f.emit({ type: "agent_start" });
      }
      await f.emit({ type: "agent_end", messages: [] });
      await f.emit({ type: "agent_end", messages: [] });
      await f.tool();
      await f.emit({ type: "agent_end", messages: [] });
      await flushTracing();
      const spans = f.capture.spans();
      assert.equal(spans.filter(span => span.name === "omp-agent").length, 3);
      assertToolParent(spans, "latest");
      assert.equal(rootFor(spans, "first")?.attrs["langfuse.observation.metadata.cancelled"], "true");
      assert.equal(rootFor(spans, "second")?.attrs["langfuse.observation.metadata.cancelled"], "true");
    } finally {
      await f.close();
    }
  });

  it("keeps automatic continuation in the original prompt trace", async () => {
    const f = await fixture();
    try {
      f.register();
      await f.prompt("continue this");
      await f.emit({ type: "agent_start" });
      await f.emit({ type: "agent_end", messages: [], willContinue: true });
      await f.emit({ type: "agent_start" });
      await f.tool();
      await f.emit({ type: "agent_end", messages: [] });
      await flushTracing();
      const spans = f.capture.spans();
      assert.equal(spans.filter(span => span.name === "omp-agent").length, 1);
      assertToolParent(spans, "continue this");
    } finally {
      await f.close();
    }
  });

  it("traces a run OMP starts without a prompt instead of dropping its generations", async () => {
    const f = await fixture();
    try {
      f.register();
      // Background results and live delegation arrive as custom messages: no before_agent_start.
      await f.emit({ type: "agent_start" });
      await f.emit({ type: "before_provider_request", payload: { model: "fixture", messages: [{ role: "user", content: "earlier prompt" }] } });
      await f.emit({
        type: "message_end",
        message: { role: "assistant", content: [{ type: "text", text: "background result handled" }], api: "openai-completions", provider: "fixture", model: "fixture", stopReason: "stop" },
      });
      await f.emit({ type: "agent_end", messages: [] });
      await flushTracing();
      const spans = f.capture.spans();
      const root = spans.find(span => span.name === "omp-agent");
      const generation = spans.find(span => span.name === "call-llm");
      assert.ok(root, "the prompt-less run must get a turn trace");
      assert.ok(generation, "the prompt-less run's generation must be exported");
      assert.equal(generation.parentSpanId, root.spanId);
      assert.equal(root.attrs["langfuse.observation.metadata.turn_trigger"], "continuation");
      assert.equal(root.attrs["langfuse.observation.metadata.session_title"], "Ordering fixture");
    } finally {
      await f.close();
    }
  });

  it("keeps a new prompt's trace open when the prompt-less run it superseded ends late", async () => {
    const f = await fixture();
    try {
      f.register();
      await f.emit({ type: "agent_start" });
      await f.emit({ type: "before_provider_request", payload: { model: "fixture", messages: [{ role: "user", content: "earlier prompt" }] } });
      await f.prompt("new prompt");
      await f.emit({ type: "agent_start" });
      // The background run's end carries no message the tracer owns.
      await f.emit({ type: "agent_end", messages: [] });
      await f.tool();
      await f.emit({ type: "agent_end", messages: [] });
      await flushTracing();
      const spans = f.capture.spans();
      assert.equal(spans.filter(span => span.name === "omp-agent").length, 2);
      assertToolParent(spans, "new prompt");
    } finally {
      await f.close();
    }
  });

  it("records omp's session-title fork as its own generation instead of superseding the reply", async () => {
    const f = await fixture();
    try {
      f.register();
      await f.prompt("hello");
      await f.emit({ type: "agent_start" });
      await f.emit({ type: "before_provider_request", payload: { model: "fixture", messages: [{ role: "user", content: "hello" }] } });
      // The title fork fires while the reply is still streaming.
      await f.emit({
        type: "before_provider_request",
        payload: { model: "tiny-fixture", messages: [{ role: "user", content: "<title-request>hello</title-request>" }] },
      });
      await f.emit({
        type: "message_end",
        message: { role: "assistant", content: [{ type: "text", text: "Hi." }], api: "openai-completions", provider: "fixture", model: "fixture", stopReason: "stop" },
      });
      await f.emit({ type: "agent_end", messages: [] });
      await flushTracing();
      const spans = f.capture.spans();
      const reply = spans.find(span => span.name === "call-llm");
      const title = spans.find(span => span.name === "generate-title");
      assert.ok(reply);
      assert.ok(title);
      assert.equal(reply.attrs["langfuse.observation.metadata.superseded"], undefined);
      assert.equal(JSON.parse(String(reply.attrs["langfuse.observation.output"])).content, "Hi.");
      assert.equal(title.attrs["langfuse.observation.model.name"], "tiny-fixture");
      assert.equal(title.attrs["langfuse.observation.output"], undefined);
    } finally {
      await f.close();
    }
  });

  it("settles a turn whose promised continuation never arrived as complete, not cancelled", async () => {
    const f = await fixture();
    try {
      f.register();
      await f.prompt("first");
      await f.emit({ type: "agent_start" });
      await f.emit({ type: "agent_end", messages: [], willContinue: true });
      await f.prompt("second");
      await f.emit({ type: "agent_start" });
      await f.emit({ type: "agent_end", messages: [] });
      await flushTracing();
      const first = rootFor(f.capture.spans(), "first");
      assert.ok(first);
      assert.equal(first.attrs["langfuse.observation.metadata.cancelled"], undefined);
      assert.equal(first.attrs["langfuse.observation.metadata.continuation_missing"], "true");
    } finally {
      await f.close();
    }
  });

  it("honors the tracing kill switch even with explicitly supplied credentials", async () => {
    const f = await fixture();
    try {
      process.env.LANGFUSE_TRACING_ENABLED = "false";
      f.register();
      await f.prompt("must not export");
      await f.emit({ type: "agent_start" });
      await f.emit({ type: "agent_end", messages: [] });
      assert.deepEqual(f.capture.requests, []);
    } finally {
      await f.close();
    }
  });
});
