import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import { z } from "@oh-my-pi/pi-coding-agent";
import { RawSseDebugBuffer } from "@oh-my-pi/pi-tui/apps/debug/raw-sse-buffer";
import { registerRawCapture } from "../src/raw-capture.ts";
import { pruneCaptures } from "../src/raw-capture-core.ts";

describe("pruneCaptures", () => {
  it("expires captures by file name at the cutoff and leaves files it does not own", async () => {
    const root = mkdtempSync(join(tmpdir(), "raw-capture-"));
    // Not JSON: retention must decide from names alone, never by reading a capture.
    const write = (dir: string, name: string) => {
      mkdirSync(join(root, dir), { recursive: true });
      writeFileSync(join(root, dir, name), "{not json");
    };
    try {
      // Cutoff is 72 hours earlier: 2026-10-04T12:00:00.000Z.
      const now = Date.parse("2026-10-07T12:00:00.000Z");
      write("anthropic/2026-10-04", "115959.999-rt-000001-request.json");
      write("anthropic/2026-10-04", "115959.999-rt-000002-request.json.0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0.tmp");
      write("anthropic/2026-10-04", "120000.000-rt-000003-message.json");
      write("anthropic/2026-10-04", "1159-rt-session-stream.jsonl");
      write("anthropic/2026-10-04", "1200-rt-session-stream.jsonl");
      write("anthropic/2026-10-04", "notes.txt");
      write("anthropic/2026-10-05", "000000.000-rt-000004-request.json");
      write("openai-codex/2026-10-03", "235959.000-rt-000005-response.json");

      const result = await pruneCaptures(root, now);

      assert.equal(result.deleted, 4);
      assert.equal(result.skipped, 1);
      assert.deepEqual(result.errors, []);
      assert.deepEqual(readdirSync(join(root, "anthropic/2026-10-04")).sort(), [
        "1200-rt-session-stream.jsonl",
        "120000.000-rt-000003-message.json",
        "notes.txt",
      ]);
      assert.deepEqual(readdirSync(join(root, "anthropic/2026-10-05")), ["000000.000-rt-000004-request.json"]);
      // An emptied day directory is removed, and the retention lock is released.
      assert.deepEqual(readdirSync(join(root, "openai-codex")), []);
      assert.deepEqual(readdirSync(root).sort(), ["anthropic", "openai-codex"]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

describe("raw capture shutdown", () => {
  it("does not replay a closed buffer and captures a replacement buffer when the session restarts", async () => {
    const root = mkdtempSync(join(tmpdir(), "raw-capture-shutdown-"));
    const savedDir = process.env.OMP_RAW_REQUESTS_DIR;
    const savedSse = process.env.OMP_RAW_REQUESTS_SSE;
    process.env.OMP_RAW_REQUESTS_DIR = root;
    process.env.OMP_RAW_REQUESTS_SSE = "1";
    const handlers = new Map<string, Array<(event: unknown, ctx: ExtensionContext) => unknown>>();
    const ticks: Array<() => void> = [];
    const firstBuffer = new RawSseDebugBuffer();
    const originalRecordEvent = firstBuffer.recordEvent;
    const manager = { getSessionId: () => "shutdown-fixture" };
    const session = { sessionManager: manager, rawSseDebugBuffer: firstBuffer };
    // Only the host boundary is controlled: the debug buffer and capture writer are real.
    const api = {
      on: (name: string, handler: (event: unknown, ctx: ExtensionContext) => unknown) =>
        handlers.set(name, [...(handlers.get(name) ?? []), handler]),
      logger: { info() { }, warn() { } },
      pi: {
        VERSION: "fixture",
        getAgentDir: () => root,
        AgentRegistry: { global: () => ({ list: () => [{ session }] }) },
      },
    } as unknown as ExtensionAPI;
    const ctx = {
      agent: { id: "ShutdownMain", kind: "main", name: "main", depth: 0 },
      sessionManager: manager, cwd: "/fixture", hasUI: false,
      getSystemPrompt: () => [],
      setInterval: (tick: () => void) => ticks.push(tick),
    } as unknown as ExtensionContext;
    const emit = async (type: string) => {
      for (const handler of handlers.get(type) ?? []) await handler({ type }, ctx);
    };
    let started = false;
    try {
      registerRawCapture(api);
      await emit("session_start");
      started = true;
      firstBuffer.recordEvent({ event: "fixture", data: 'data: {"marker":"first"}'.slice(6), raw: ['data: {"marker":"first"}'] });
      await emit("session_shutdown");
      started = false;
      assert.equal(firstBuffer.recordEvent, originalRecordEvent);
      for (const tick of ticks) tick();
      assert.equal(firstBuffer.recordEvent, originalRecordEvent, "a late sweep must not reattach to the still-registered closed session");
      firstBuffer.recordEvent({ event: "fixture", data: 'data: {"marker":"after shutdown"}'.slice(6), raw: ['data: {"marker":"after shutdown"}'] });
      for (const tick of ticks) tick();

      session.rawSseDebugBuffer = new RawSseDebugBuffer();
      await emit("session_start");
      started = true;
      session.rawSseDebugBuffer.recordEvent({ event: "fixture", data: 'data: {"marker":"second"}'.slice(6), raw: ['data: {"marker":"second"}'] });
      await emit("session_shutdown");
      started = false;
      for (const tick of ticks) tick();

      const stream = z.object({
        kind: z.literal("stream"),
        sessionId: z.string(),
        captureStage: z.string(),
        record: z.object({ raw: z.array(z.string()) }),
      });
      const files = readdirSync(root, { encoding: "utf8", recursive: true }).sort();
      const frames = files.filter(file => file.endsWith("-stream.jsonl"))
        .flatMap(file => readFileSync(join(root, file), "utf8").trim().split("\n"))
        .map(line => stream.parse(JSON.parse(line)));
      assert.deepEqual(frames.map(frame => frame.record.raw), [
        ['data: {"marker":"first"}'],
        ['data: {"marker":"second"}'],
      ]);
      assert.ok(frames.every(frame => frame.captureStage === "before-debug-buffer-truncation"));
      assert.deepEqual(files.filter(file => file.endsWith("-gap.json")), []);
    } finally {
      if (started) await emit("session_shutdown");
      if (savedDir === undefined) delete process.env.OMP_RAW_REQUESTS_DIR;
      else process.env.OMP_RAW_REQUESTS_DIR = savedDir;
      if (savedSse === undefined) delete process.env.OMP_RAW_REQUESTS_SSE;
      else process.env.OMP_RAW_REQUESTS_SSE = savedSse;
      rmSync(root, { recursive: true, force: true });
    }
  });
});

describe("raw capture correlation", () => {
  it("retires an unanswered request when OMP re-sends the same history, so later replies stay matched", async () => {
    const root = mkdtempSync(join(tmpdir(), "raw-capture-resend-"));
    const savedDir = process.env.OMP_RAW_REQUESTS_DIR;
    const savedSse = process.env.OMP_RAW_REQUESTS_SSE;
    process.env.OMP_RAW_REQUESTS_DIR = root;
    process.env.OMP_RAW_REQUESTS_SSE = "0";
    const handlers = new Map<string, Array<(event: unknown, ctx: ExtensionContext) => unknown>>();
    const manager = { getSessionId: () => "resend-fixture", getSessionName: () => "Resend fixture" };
    const session = { sessionManager: manager, rawSseDebugBuffer: new RawSseDebugBuffer(), agent: { state: { messages: [] } } };
    const api = {
      on: (name: string, handler: (event: unknown, ctx: ExtensionContext) => unknown) =>
        handlers.set(name, [...(handlers.get(name) ?? []), handler]),
      logger: { info() { }, warn() { } },
      pi: {
        VERSION: "fixture",
        getAgentDir: () => root,
        AgentRegistry: { global: () => ({ list: () => [{ session }] }) },
      },
    } as unknown as ExtensionAPI;
    const ctx = {
      agent: { id: "ResendMain", kind: "main", name: "main", depth: 0 },
      sessionManager: manager, cwd: "/fixture", hasUI: false,
      getSystemPrompt: () => [],
      setInterval: () => { },
      model: { provider: "ollama-cloud", id: "kimi-k3", api: "openai-completions", baseUrl: "https://ollama.com/v1" },
    } as unknown as ExtensionContext;
    const emit = async (type: string, event: Record<string, unknown> = {}) => {
      for (const handler of handlers.get(type) ?? []) await handler({ type, ...event }, ctx);
    };
    const reply = (cacheRead: number) => ({
      role: "assistant", provider: "ollama-cloud", model: "kimi-k3", api: "openai-completions", stopReason: "toolUse",
      content: [], usage: { input: 10, output: 5, cacheRead, cacheWrite: 0 }, responseId: `chatcmpl-${cacheRead}`,
    });
    let started = false;
    try {
      registerRawCapture(api);
      await emit("session_start");
      started = true;
      const history = [{ role: "user", content: "task" }];
      // The first attempt fails without reaching extensions; OMP sends the same history again.
      await emit("before_provider_request", { payload: { model: "kimi-k3", messages: [...history] } });
      await emit("before_provider_request", { payload: { model: "kimi-k3", messages: [...history] } });
      await emit("message_end", { message: reply(1) });
      history.push({ role: "assistant", content: "next" });
      await emit("before_provider_request", { payload: { model: "kimi-k3", messages: [...history] } });
      await emit("message_end", { message: reply(2) });
      await emit("session_shutdown");
      started = false;

      const record = z.object({
        kind: z.string(),
        eventSequence: z.number(),
        sessionTitle: z.string().optional(),
        captureId: z.string().optional(),
        reason: z.string().optional(),
        captureIds: z.array(z.string()).optional(),
        supersededBy: z.string().optional(),
        correlation: z.object({ method: z.string(), captureId: z.string().optional() }).optional(),
      });
      const records = readdirSync(root, { encoding: "utf8", recursive: true })
        .filter(file => file.endsWith(".json"))
        .map(file => record.parse(JSON.parse(readFileSync(join(root, file), "utf8"))))
        .sort((a, b) => a.eventSequence - b.eventSequence);
      const requests = records.filter(r => r.kind === "request");
      const messages = records.filter(r => r.kind === "message");
      assert.equal(requests.length, 3);
      assert.deepEqual(messages.map(m => m.correlation), [
        { method: "single-flight", captureId: requests[1]!.captureId },
        { method: "single-flight", captureId: requests[2]!.captureId },
      ]);
      assert.deepEqual(
        records.filter(r => r.kind === "gap").map(gap => [gap.reason, gap.captureIds, gap.supersededBy]),
        [["attempt-superseded-without-response", [requests[0]!.captureId], requests[1]!.captureId]],
      );
      assert.ok([...requests, ...messages].every(r => r.sessionTitle === "Resend fixture"));
    } finally {
      if (started) await emit("session_shutdown");
      if (savedDir === undefined) delete process.env.OMP_RAW_REQUESTS_DIR;
      else process.env.OMP_RAW_REQUESTS_DIR = savedDir;
      if (savedSse === undefined) delete process.env.OMP_RAW_REQUESTS_SSE;
      else process.env.OMP_RAW_REQUESTS_SSE = savedSse;
      rmSync(root, { recursive: true, force: true });
    }
  });
});
