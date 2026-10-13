import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, symlinkSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { REPO_ROOT, startCaptureServer, type Capture, type CapturedSpan } from "./helpers.ts";
import type { BackfillReport } from "../src/backfill.ts";
import { enumerateCaptureFiles, readCapture, snapshotCaptures } from "../src/backfill/archive.ts";
import { pruneCaptures } from "../src/raw-capture-core.ts";

const SECRET = "sk-lf-replay-fixture-secret";
const BASE = Date.parse("2026-10-04T12:00:00Z");
const model = { provider: "mock", id: "replay-model", api: "openai-completions" };
const usage = { input: 100, output: 20, cacheRead: 80, cacheWrite: 10, reasoningTokens: 5,
  cost: { input: 0.1, output: 0.2, cacheRead: 0.03, cacheWrite: 0.04, total: 0.37 } };
function assistant(content: unknown[], stopReason = "stop") {
  return { role: "assistant", content, api: model.api, provider: model.provider, model: model.id, usage, stopReason, ttft: 5, duration: 20 };
}
function archive(root: string, nested = false) {
  let sequence = 0;
  const sources: string[] = [];
  const add = (kind: string, data: Record<string, unknown>, sessionId = "archived-session") => {
    const ts = new Date(BASE + ++sequence * 100).toISOString();
    const name = `${ts.slice(11, 23).replaceAll(":", "")}-fixture-runtime-${String(sequence).padStart(6, "0")}-${kind}.json`;
    const directory = join(root, "mock", ts.slice(0, 10));
    mkdirSync(directory, { recursive: true });
    const file = join(directory, name);
    writeFileSync(file, JSON.stringify({ schemaVersion: 2, kind, ts, eventSequence: sequence,
      runtime: { id: "fixture-runtime" }, sessionId, cwd: "/archived/workspace", agent: { id: "Main", kind: "main" }, model, ...data }));
    sources.push(file);
    return file;
  };
  const tool = { id: "call_real", type: "function", function: { name: "bash", arguments: '{"command":"pwd"}' } };
  const history = [{ role: "user", content: `inspect ${SECRET}` }];
  const payload = (messages: unknown[]) => ({ model: model.id, max_tokens: 1024, messages: [{ role: "system", content: "be accurate" }, ...messages],
    tools: [{ type: "function", function: { name: "bash", description: "run command", parameters: { type: "object" } } }] });
  add("lifecycle", { event: { type: "turn_start", turnIndex: 0 } });
  const first = add("request", { captureId: "first", payload: payload(history) });
  add("wire_request", { payload: payload(history) });
  add("message", { correlation: { method: "single-flight", captureId: "first" },
    message: assistant([{ type: "toolCall", id: "call_real", name: "bash", arguments: { command: "pwd" } }], "toolUse") });
  if (nested) {
    const agent = { id: "Worker", kind: "sub", parentId: "Main" };
    add("request", { captureId: "child", agent, payload: payload([{ role: "user", content: "child inspection" }]) }, "child-session");
    add("message", { agent, correlation: { method: "single-flight", captureId: "child" },
      message: assistant([{ type: "text", text: "child complete" }]) }, "child-session");
    add("lifecycle", { agent, event: { type: "session_shutdown" } }, "child-session");
  }
  add("lifecycle", { event: { type: "turn_start", turnIndex: 4 } });
  add("request", { captureId: "second", payload: payload([...history,
    { role: "assistant", tool_calls: [tool] }, { role: "tool", tool_call_id: "call_real", content: "/archived/workspace" }]) });
  add("message", { correlation: { method: "response-id", captureId: "second" },
    message: assistant([{ type: "thinking", thinking: "check the result" }, { type: "text", text: "complete" }]) });
  add("lifecycle", { event: { type: "session_shutdown" } });
  add("message", { correlation: { method: "ambiguous", candidates: ["pruned-a", "pruned-b"] },
    message: assistant([{ type: "text", text: "orphan output retained" }]) }, "orphan-session");
  add("request", { captureId: "delta", model: { ...model, api: "openai-codex-responses" }, payload: {
    type: "response.create", previous_response_id: "pruned-response", instructions: "continue carefully",
    input: [{ type: "function_call_output", call_id: "call_pruned", output: "retained tool output" }], tools: [] } }, "delta-session");
  add("message", { correlation: { method: "single-flight", captureId: "delta" },
    message: { ...assistant([{ type: "text", text: "delta answer retained" }]), api: "openai-codex-responses" } }, "delta-session");
  const bytes = sources.map(file => readFileSync(file));
  return { first, sources, bytes, add, payload };
}
interface Result { code: number | null; stdout: string; stderr: string }
function command(root: string, capture?: Capture, args: string[] = [], env: Record<string, string | undefined> = {}): Promise<Result> {
  const { promise, resolve, reject } = Promise.withResolvers<Result>();
    const child = spawn(process.execPath, [join(REPO_ROOT, "src/backfill.ts"), "--root", root, ...args], { cwd: REPO_ROOT, env: {
      ...process.env, OMP_PROFILE: "", PI_PROFILE: "", HOME: root, OMP_CODING_AGENT_DIR: root,
      LANGFUSE_PUBLIC_KEY: capture ? "pk-lf-replay-fixture" : undefined, LANGFUSE_SECRET_KEY: capture ? SECRET : undefined,
      LANGFUSE_BASE_URL: capture ? `http://127.0.0.1:${capture.port}` : undefined,
      LANGFUSE_TRACING_ENABLED: "true", LANGFUSE_MEDIA_UPLOAD_ENABLED: "false", ...env }, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    child.stdout.on("data", data => { stdout += String(data); });
    child.stderr.on("data", data => { stderr += String(data); });
    child.once("error", reject);
    child.once("close", code => resolve({ code, stdout, stderr }));
  return promise;
}
function decoded(span: CapturedSpan, key: string) {
  const raw = span.attrs[`langfuse.observation.${key}`];
  return raw === undefined ? undefined : JSON.parse(String(raw));
}
describe("raw archive replay through the real tracing SDK", () => {
  it("retains generations, inferred tools and orphan facts with historical IDs, times and redaction", async () => {
    const root = mkdtempSync(join(tmpdir(), "omp-backfill-"));
    const fixture = archive(root);
    const capture = await startCaptureServer();
    try {
      const result = await command(root, capture);
      assert.equal(result.code, 0, result.stderr);
      const report: BackfillReport = JSON.parse(result.stdout);
      assert.equal(report.requests, 3);
      assert.equal(report.unsupported, 0);
      assert.ok(report.unmatched > 0);
      const spans = capture.spans();
      const roots = spans.filter(span => span.name === "omp-agent");
      assert.equal(roots.length, 3, "turnIndex 4 is a model step, not another prompt root");
      assert.deepEqual(new Set(roots.map(span => span.traceId)), new Set(report.rootIds));
      const generations = spans.filter(span => span.name === "call-llm");
      assert.equal(generations.length, 4, "orphans and delta-only requests must not be discarded");
      assert.ok(generations.every(span => span.startNs >= BigInt(BASE) * 1_000_000n && span.endNs < BigInt(BASE + 5000) * 1_000_000n));
      const orphan = generations.find(span => decoded(span, "output")?.content === "orphan output retained")!;
      assert.ok(orphan);
      assert.equal(orphan.endNs - orphan.startNs, 20_000_000n, "an orphan's archived duration determines its inferred interval");
      assert.equal(decoded(orphan, "input"), undefined, "an orphan must not inherit another request's input");
      assert.equal(String(orphan.attrs["langfuse.observation.metadata.request_missing"]), "true");
      assert.deepEqual(decoded(orphan, "usage_details"), { input: 100, output: 15, output_reasoning_tokens: 5, cache_read_input_tokens: 80, cache_creation_input_tokens: 10 });
      const tool = spans.find(span => span.name === "bash")!;
      assert.ok(tool);
      assert.equal(tool.attrs["langfuse.observation.output"], "/archived/workspace");
      assert.ok(tool.endNs > tool.startNs);
      const toolContext = generations.flatMap(span => decoded(span, "input") ?? [])
        .find(message => message.role === "tool" && message.tool_call_id === "call_real");
      assert.equal(toolContext?.name, "bash", "paired wire history and pending records must recover the same actual tool name");
      const parent = roots.find(span => span.spanId === tool.parentSpanId)!;
      assert.ok(parent && parent.endNs >= tool.endNs);
      assert.equal(JSON.stringify(capture.requests).includes(SECRET), false);
      assert.ok(generations.some(span => JSON.stringify(decoded(span, "input"))?.includes("retained tool output")));
      const identities = spans.map(span => `${span.traceId}/${span.spanId}`).sort();
      const priorLength = spans.length;
      assert.equal((await command(root, capture)).code, 0);
      assert.deepEqual(capture.spans().slice(priorLength).map(span => `${span.traceId}/${span.spanId}`).sort(), identities,
        "a retry must upsert the same historical observations, not create duplicates");
      for (const [index, file] of fixture.sources.entries()) assert.deepEqual(readFileSync(file), fixture.bytes[index]);
      const recoveryFile = join(root, "recover.json");
      const targetId = orphan.spanId;
      writeFileSync(recoveryFile, JSON.stringify([targetId]));
      const beforeRecovery = capture.spans().length;
      const recovered = await command(root, capture, ["--only-observations", recoveryFile]);
      assert.equal(recovered.code, 0, recovered.stderr);
      assert.deepEqual(capture.spans().slice(beforeRecovery).map(span => span.spanId), [targetId],
        "targeted recovery must retain the original SDK identity without replaying accepted observations");
    } finally { capture.close(); rmSync(root, { recursive: true, force: true }); }
  });
  it("reports the real inherited trace IDs of replayed in-process subagents", async () => {
    const root = mkdtempSync(join(tmpdir(), "omp-backfill-parent-"));
    archive(root, true);
    const capture = await startCaptureServer();
    try {
      const result = await command(root, capture);
      assert.equal(result.code, 0, result.stderr);
      const report: BackfillReport = JSON.parse(result.stdout);
      const spans = capture.spans();
      const child = spans.find(span => span.name === "omp-subagent-agent")!;
      assert.ok(child, "the real SDK must produce the archived subagent");
      const parent = spans.find(span => span.spanId === child.parentSpanId)!;
      assert.ok(parent, "the real SDK must resolve its in-process parent");
      assert.equal(child.traceId, parent.traceId);
      assert.deepEqual(new Set(report.rootIds), new Set(spans.map(span => span.traceId)));
    } finally { capture.close(); rmSync(root, { recursive: true, force: true }); }
  });
  it("replays a protected input snapshot after real retention deletes the source captures", async () => {
    const root = mkdtempSync(join(tmpdir(), "omp-backfill-retention-"));
    const fixture = archive(root);
    const capture = await startCaptureServer();
    let snapshotRoot: string | undefined;
    try {
      const startup = Date.now();
      const files = await enumerateCaptureFiles(root, startup, undefined, Date.now(), { skipped: 0, incomplete: 0 });
      assert.equal(files.length, fixture.sources.length, JSON.stringify({ startup, times: fixture.sources.map(file => {
        const stat = statSync(file);
        return { birthtime: stat.birthtimeMs, ctime: stat.ctimeMs, mtime: stat.mtimeMs };
      }) }));
      const snapshot = await snapshotCaptures(files, () => { throw new Error("fixture unexpectedly disappeared"); }, { onSkip() {} });
      snapshotRoot = snapshot.root;
      const original = await readCapture(snapshot.files[0]!);
      const removed = await pruneCaptures(root, BASE + 4 * 86_400_000);
      assert.equal(removed.deleted, fixture.sources.length);
      assert.deepEqual(await readCapture(snapshot.files[0]!), original);
      const result = await command(snapshot.root, capture);
      assert.equal(result.code, 0, result.stderr);
      assert.equal(capture.spans().filter(span => span.name === "call-llm").length, 4,
        JSON.stringify({ report: JSON.parse(result.stdout), spans: capture.spans().map(span => ({ id: span.spanId, name: span.name })) }));
    } finally {
      capture.close();
      if (snapshotRoot) rmSync(snapshotRoot, { recursive: true, force: true });
      rmSync(root, { recursive: true, force: true });
    }
  });
  it("drains many unfinished prompt roots without exhausting the SDK export concurrency", async () => {
    const root = mkdtempSync(join(tmpdir(), "omp-backfill-drain-"));
    const fixture = archive(root);
    const capture = await startCaptureServer();
    try {
      for (let index = 0; index < 80; index++) {
        const captureId = `unfinished-${index}`;
        const sessionId = `unfinished-session-${index}`;
        fixture.add("request", { captureId, payload: fixture.payload([{ role: "user", content: `prompt ${index}` }]) }, sessionId);
        fixture.add("message", { correlation: { method: "single-flight", captureId },
          message: assistant([{ type: "text", text: `answer ${index}` }]) }, sessionId);
      }
      const result = await command(root, capture, [], { LANGFUSE_FLUSH_AT: "1" });
      assert.equal(result.code, 0, result.stderr);
      const report: BackfillReport = JSON.parse(result.stdout);
      assert.deepEqual(new Set(capture.spans().map(span => span.spanId)), new Set(report.observationIds));
      assert.equal(capture.spans().filter(span => span.name === "omp-agent").length, 83);
    } finally { capture.close(); rmSync(root, { recursive: true, force: true }); }
  });
  it("dry-runs without credentials, filters sessions and rejects disabled export and invalid options", async () => {
    const root = mkdtempSync(join(tmpdir(), "omp-backfill-policy-"));
    archive(root);
    try {
      const result = await command(root, undefined, ["--dry-run", "--session", "delta-session"]);
      assert.equal(result.code, 0, result.stderr);
      const report: BackfillReport = JSON.parse(result.stdout);
      assert.equal(report.requests, 1);
      assert.equal(report.imported, 0);
      assert.ok(report.wouldImport > 0);
      const disabled = await command(root, undefined, [], { LANGFUSE_TRACING_ENABLED: "false" });
      assert.notEqual(disabled.code, 0);
      assert.match(disabled.stderr, /disabled by LANGFUSE_TRACING_ENABLED/);
      assert.notEqual((await command(root, undefined, ["--dry-run", "--limit", "0"])).code, 0);
      assert.notEqual((await command(root, undefined, ["--dry-run", "--from", "not-a-time"])).code, 0);
      const link = join(root, "linked-root"); symlinkSync(join(root, "mock"), link);
      assert.notEqual((await command(link, undefined, ["--dry-run"])).code, 0);
    } finally { rmSync(root, { recursive: true, force: true }); }
  });
});
