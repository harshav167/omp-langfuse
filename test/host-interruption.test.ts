import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { createInterface } from "node:readline";
import { describe, it } from "node:test";
import type { Capture, CapturedSpan } from "./helpers.ts";
import { REPO_ROOT, startCaptureServer } from "./helpers.ts";

const INTERRUPTED_PROMPT = "interrupt this real host run";
const REPLACEMENT_PROMPT = "replacement prompt must own its output";
const TOOL_OUTPUT = "replacement-tool-output";
const FINAL_OUTPUT = "replacement final answer after interruption";
const DEADLINE_MS = 20_000;
const CHILD_DEADLINE_MS = 60_000;

interface ProviderRequest {
  body: Record<string, unknown>;
  response: ServerResponse;
}

interface LocalHostHarness {
  port: number;
  requests: ProviderRequest[];
  firstRequest: Promise<ProviderRequest>;
  firstResponseClosed: Promise<void>;
  replacementFollowup: Promise<ProviderRequest>;
  replacementFinal: Promise<ProviderRequest>;
  oldEndGateEntered: Promise<number>;
  oldEndGateHeld: () => boolean;
  releaseOldEndGate: () => void;
  completeReplacement: () => void;
  close: () => Promise<void>;
}

interface DriverEvent {
  type: string;
  [key: string]: unknown;
}

interface Driver {
  child: ChildProcess;
  send: (value: Record<string, unknown>) => void;
  waitFor: (type: string, predicate?: (event: DriverEvent) => boolean, timeoutMs?: number) => Promise<DriverEvent>;
  waitForExit: () => Promise<{ code: number | null; signal: NodeJS.Signals | null }>;
  events: () => readonly DriverEvent[];
  stderr: () => string;
}

function writeSseEvent(response: ServerResponse, body: unknown): void {
  response.write(`data: ${JSON.stringify(body)}\n\n`);
}

function writeOpenAIHeaders(response: ServerResponse): void {
  response.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
}


function requestBody(request: IncomingMessage): Promise<Record<string, unknown>> {
  return (async () => {
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(Buffer.from(chunk));
    const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("Local host harness expected a JSON object request");
    }
    return parsed as Record<string, unknown>;
  })();
}

function sendToolCall(response: ServerResponse, model: string): void {
  const base = { id: "chatcmpl-replacement-tool", object: "chat.completion.chunk", created: 1, model };
  writeSseEvent(response, {
    ...base,
    choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
  });
  writeSseEvent(response, {
    ...base,
    choices: [{
      index: 0,
      delta: {
        tool_calls: [{
          index: 0,
          id: "call_replacement_tool",
          type: "function",
          function: {
            name: "bash",
            arguments: JSON.stringify({ command: `printf '${TOOL_OUTPUT}\\n'` }),
          },
        }],
      },
      finish_reason: null,
    }],
  });
  writeSseEvent(response, { ...base, choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] });
  writeSseEvent(response, {
    ...base,
    choices: [],
    usage: { prompt_tokens: 48, completion_tokens: 12, total_tokens: 60 },
  });
  response.end("data: [DONE]\n\n");
}

function sendFinalAnswer(response: ServerResponse, model: string): void {
  const base = { id: "chatcmpl-replacement-final", object: "chat.completion.chunk", created: 2, model };
  writeSseEvent(response, {
    ...base,
    choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
  });
  writeSseEvent(response, {
    ...base,
    choices: [{ index: 0, delta: { content: FINAL_OUTPUT }, finish_reason: null }],
  });
  writeSseEvent(response, { ...base, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
  writeSseEvent(response, {
    ...base,
    choices: [],
    usage: { prompt_tokens: 72, completion_tokens: 18, total_tokens: 90 },
  });
  response.end("data: [DONE]\n\n");
}

async function startLocalHostHarness(): Promise<LocalHostHarness> {
  const firstRequest = Promise.withResolvers<ProviderRequest>();
  const firstResponseClosed = Promise.withResolvers<void>();
  const replacementFollowup = Promise.withResolvers<ProviderRequest>();
  const replacementFinal = Promise.withResolvers<ProviderRequest>();
  const oldEndGateEntered = Promise.withResolvers<number>();
  const requests: ProviderRequest[] = [];
  let gateResponse: ServerResponse | undefined;
  let gateReleased = false;
  const openResponses = new Set<ServerResponse>();

  const server = createServer((request, response) => {
    openResponses.add(response);
    response.once("close", () => openResponses.delete(response));
    void (async () => {
      const url = new URL(request.url ?? "/", "http://127.0.0.1");
      if (request.method === "POST" && url.pathname === "/__control/delay-old-agent-end") {
        const body = await requestBody(request);
        const sequence = body.sequence;
        if (sequence !== 1) throw new Error(`Unexpected delayed agent_end sequence: ${String(sequence)}`);
        gateResponse = response;
        oldEndGateEntered.resolve(1);
        return;
      }
      if (request.method !== "POST" || url.pathname !== "/v1/chat/completions") {
        response.writeHead(404).end();
        return;
      }

      const body = await requestBody(request);
      const entry = { body, response };
      requests.push(entry);
      const model = typeof body.model === "string" ? body.model : "host-interruption-mock";
      if (requests.length === 1) {
        writeOpenAIHeaders(response);
        response.once("close", () => firstResponseClosed.resolve());
        const base = { id: "chatcmpl-interrupted", object: "chat.completion.chunk", created: 0, model };
        writeSseEvent(response, {
          ...base,
          choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
        });
        writeSseEvent(response, {
          ...base,
          choices: [{ index: 0, delta: { content: "partial output before the user interrupt" }, finish_reason: null }],
        });
        firstRequest.resolve(entry);
        return;
      }
      if (requests.length === 2) {
        writeOpenAIHeaders(response);
        replacementFollowup.resolve(entry);
        sendToolCall(response, model);
        return;
      }
      if (requests.length === 3) {
        writeOpenAIHeaders(response);
        replacementFinal.resolve(entry);
        return;
      }
      response.writeHead(500, { "content-type": "text/plain" });
      response.end(`Unexpected provider request ${requests.length}`);
    })().catch(error => {
      if (!response.headersSent) response.writeHead(500, { "content-type": "text/plain" });
      response.end(String(error));
    });
  });

  const listening = Promise.withResolvers<number>();
  server.once("error", listening.reject);
  server.listen(0, "127.0.0.1", () => {
    const address = server.address();
    if (!address || typeof address === "string") {
      listening.reject(new Error("Mock provider did not bind a TCP port"));
      return;
    }
    listening.resolve(address.port);
  });
  const port = await listening.promise;

  return {
    port,
    requests,
    firstRequest: firstRequest.promise,
    firstResponseClosed: firstResponseClosed.promise,
    replacementFollowup: replacementFollowup.promise,
    replacementFinal: replacementFinal.promise,
    oldEndGateEntered: oldEndGateEntered.promise,
    oldEndGateHeld: () => gateResponse !== undefined && !gateReleased,
    releaseOldEndGate: () => {
      if (gateReleased) return;
      gateReleased = true;
      if (!gateResponse || gateResponse.writableEnded) return;
      gateResponse.writeHead(200, { "content-type": "application/json" });
      gateResponse.end(JSON.stringify({ released: true }));
    },
    completeReplacement: () => {
      const entry = requests[2];
      if (!entry || entry.response.writableEnded) throw new Error("Replacement final request is not waiting");
      sendFinalAnswer(entry.response, String(entry.body.model ?? "host-interruption-mock"));
    },
    close: async () => {
      if (!gateReleased) {
        gateReleased = true;
        if (gateResponse && !gateResponse.writableEnded) gateResponse.end(JSON.stringify({ released: true }));
      }
      for (const response of openResponses) {
        if (!response.writableEnded) response.end("data: [DONE]\n\n");
      }
      const closed = Promise.withResolvers<void>();
      server.close(error => error ? closed.reject(error) : closed.resolve());
      server.closeAllConnections();
      await closed.promise;
    },
  };
}

function parseDriverEvent(value: unknown): DriverEvent {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    !("type" in value) ||
    typeof value.type !== "string"
  ) {
    throw new Error("Host SDK driver emitted an invalid event");
  }
  return { ...value, type: value.type };
}
function startDriver(agentDir: string, cwd: string, providerPort: number, capturePort: number, gateUrl: string): Driver {
  const rawRequestsDir = join(agentDir, "raw-requests");
  mkdirSync(rawRequestsDir, { recursive: true });
  const env: NodeJS.ProcessEnv = {
    PATH: process.env.PATH,
    TMPDIR: tmpdir(),
    LANG: process.env.LANG ?? "en_US.UTF-8",
    HOME: agentDir,
    OMP_PROFILE: "",
    PI_PROFILE: "",
    OMP_CODING_AGENT_DIR: agentDir,
    PI_CODING_AGENT_DIR: agentDir,
    OMP_RAW_REQUESTS_DIR: rawRequestsDir,
    PI_RAW_REQUESTS_DIR: rawRequestsDir,
    PI_NO_TITLE: "1",
    PI_SKIP_VERSION_CHECK: "1",
    PI_TELEMETRY: "0",
    LANGFUSE_PUBLIC_KEY: "pk-lf-host-interruption-test",
    LANGFUSE_SECRET_KEY: "sk-lf-host-interruption-test",
    LANGFUSE_BASE_URL: `http://127.0.0.1:${capturePort}`,
    LANGFUSE_TRACING_ENABLED: "true",
    LANGFUSE_MEDIA_UPLOAD_ENABLED: "false",
    HOST_INTERRUPT_CWD: cwd,
    HOST_INTERRUPT_AGENT_DIR: agentDir,
    HOST_INTERRUPT_PROVIDER_PORT: String(providerPort),
    HOST_INTERRUPT_GATE_URL: gateUrl,
  };
  const driverPath = join(REPO_ROOT, "test", "fixtures", "interrupt-driver.ts");
  const child = spawn(process.execPath, [driverPath], {
    cwd,
    env,
    stdio: ["pipe", "pipe", "pipe"],
  });
  const events: DriverEvent[] = [];
  const waiters: Array<{
    type: string;
    predicate: (event: DriverEvent) => boolean;
    resolve: (event: DriverEvent) => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }> = [];
  let stderr = "";
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", value => { stderr += String(value); });

  const failWaiters = (error: Error): void => {
    for (const waiter of waiters.splice(0)) {
      clearTimeout(waiter.timer);
      waiter.reject(error);
    }
  };
  const input = createInterface({ input: child.stdout, crlfDelay: Infinity });

  input.on("line", line => {
    let event: DriverEvent;
    try {
      const parsed: unknown = JSON.parse(line);
      event = parseDriverEvent(parsed);
    } catch (error) {
      failWaiters(new Error(`Driver emitted invalid JSON event: ${String(error)}\n${stderr}`));
      return;
    }
    const waiterIndex = waiters.findIndex(waiter => waiter.type === event.type && waiter.predicate(event));
    if (waiterIndex < 0) events.push(event);
    else {
      const [waiter] = waiters.splice(waiterIndex, 1);
      clearTimeout(waiter!.timer);
      waiter!.resolve(event);
    }
  });
  const exited = Promise.withResolvers<{ code: number | null; signal: NodeJS.Signals | null }>();
  child.once("error", error => {
    failWaiters(error);
    exited.reject(error);
  });
  child.once("close", (code, signal) => {
    failWaiters(new Error(`Host SDK driver exited (${code ?? signal})\n${stderr}`));
    exited.resolve({ code, signal });
  });

  return {
    child,
    send(value) {
      if (!child.stdin.writable) throw new Error("Host SDK driver stdin is closed");
      child.stdin.write(`${JSON.stringify(value)}\n`);
    },
    waitFor(type, predicate = () => true, timeoutMs = DEADLINE_MS) {
      const existingIndex = events.findIndex(event => event.type === type && predicate(event));
      if (existingIndex >= 0) return Promise.resolve(events.splice(existingIndex, 1)[0]!);
      const { promise, resolve, reject } = Promise.withResolvers<DriverEvent>();
      const timer = setTimeout(() => {
        const index = waiters.findIndex(waiter => waiter.resolve === resolve);
        if (index >= 0) waiters.splice(index, 1);
        reject(new Error(`Timed out waiting for host SDK driver event ${type}\n${stderr}`));
      }, timeoutMs);
      waiters.push({ type, predicate, resolve, reject, timer });
      return promise;
    },
    waitForExit: () => exited.promise,
    stderr: () => stderr,
    events: () => events,
  };
}

function withDeadline<T>(promise: Promise<T>, description: string, timeoutMs = DEADLINE_MS): Promise<T> {
  const { promise: deadline, reject } = Promise.withResolvers<T>();
  const timer = setTimeout(() => reject(new Error(`Timed out waiting for ${description}`)), timeoutMs);
  return Promise.race([promise, deadline]).finally(() => clearTimeout(timer));
}

function parseSpanPayload(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

function contentOf(value: unknown): string {
  const parsed = parseSpanPayload(value);
  if (typeof parsed === "string") return parsed;
  if (!parsed || typeof parsed !== "object" || !("content" in parsed)) return "";
  const content = parsed.content;
  return typeof content === "string" ? content : "";
}

function promptOf(root: CapturedSpan): string {
  return contentOf(root.attrs["langfuse.observation.input"]);
}

function toolOutputWasSent(request: ProviderRequest): boolean {
  const messages = request.body.messages;
  if (!Array.isArray(messages)) return false;
  return messages.some(message => {
    if (!message || typeof message !== "object" || Array.isArray(message) || !("role" in message)) return false;
    const content = "content" in message ? message.content : undefined;
    return message.role === "tool" && typeof content === "string" && content.includes(TOOL_OUTPUT);
  });
}


describe("real OMP host prompt interruption tracing", () => {
  it("keeps delayed interrupted agent_end from dropping replacement generations and tool output", { timeout: CHILD_DEADLINE_MS }, async () => {
    let capture: Capture | undefined;
    let host: LocalHostHarness | undefined;
    let driver: Driver | undefined;
    let agentDir: string | undefined;
    let cwd: string | undefined;
    let shutdownSent = false;
    try {
      const activeCapture = await startCaptureServer();
      capture = activeCapture;
      const activeHost = await startLocalHostHarness();
      host = activeHost;
      agentDir = mkdtempSync(join(tmpdir(), "omp-host-interrupt-agent-"));
      cwd = mkdtempSync(join(tmpdir(), "omp-host-interrupt-workspace-"));
      const gateUrl = `http://127.0.0.1:${activeHost.port}/__control/delay-old-agent-end`;
      const activeDriver = startDriver(agentDir, cwd, activeHost.port, activeCapture.port, gateUrl);
      driver = activeDriver;

      const ready = await activeDriver.waitFor("ready");
      assert.equal(ready.target, "installed OMP createAgentSession + AgentSession");
      assert.deepEqual(ready.capabilities, [
        "session.prompt",
        "session.abort",
        "real extension lifecycle events",
        "real tool execution",
      ]);
      const reportedOptions = ready.options;
      if (!reportedOptions || typeof reportedOptions !== "object" || Array.isArray(reportedOptions)) {
        throw new Error("SDK driver omitted its actual session options");
      }
      if (!("extensionOrder" in reportedOptions) || !Array.isArray(reportedOptions.extensionOrder)) {
        throw new Error("SDK driver omitted the loaded extension order");
      }
      assert.deepEqual(reportedOptions.extensionOrder.slice(0, 3), [
        "delay-first-agent_end",
        "langfuse-tracing",
        "post-tracing-agent_end-observer",
      ]);
      if (
        !("disableExtensionDiscovery" in reportedOptions) || reportedOptions.disableExtensionDiscovery !== true ||
        !("enableMCP" in reportedOptions) || reportedOptions.enableMCP !== false ||
        !("enableLsp" in reportedOptions) || reportedOptions.enableLsp !== false
      ) {
        throw new Error("SDK driver did not disable ambient extensions, MCP, and LSP discovery");
      }
      if (!("skills" in reportedOptions) || !Array.isArray(reportedOptions.skills)) {
        throw new Error("SDK driver omitted its explicit empty skills configuration");
      }
      if (!("rules" in reportedOptions) || !Array.isArray(reportedOptions.rules)) {
        throw new Error("SDK driver omitted its explicit empty rules configuration");
      }
      assert.deepEqual(reportedOptions.skills, []);
      assert.deepEqual(reportedOptions.rules, []);
      if (!("providerBaseUrl" in reportedOptions) || typeof reportedOptions.providerBaseUrl !== "string") {
        throw new Error("SDK driver omitted the mock provider endpoint");
      }
      assert.equal(reportedOptions.providerBaseUrl, `http://127.0.0.1:${activeHost.port}/v1`);


      activeDriver.send({ id: "old-prompt", type: "prompt", label: "interrupted", prompt: INTERRUPTED_PROMPT });
      await activeDriver.waitFor("prompt_called", event => event.id === "old-prompt");
      const oldProviderRequest = await withDeadline(activeHost.firstRequest, "first real provider request").catch(error => {
        throw new Error(`${String(error)}\n${activeDriver.stderr()}\n${JSON.stringify(activeDriver.events())}`);
      });
      assert.equal(oldProviderRequest.body.stream, true, "the installed provider adapter must stream over HTTP");
      await activeDriver.waitFor("provider_text_delta_observed");

      activeDriver.send({ id: "abort-old", type: "abort" });
      const [, , abortEvent] = await withDeadline(Promise.all([
        activeHost.firstResponseClosed,
        activeHost.oldEndGateEntered,
        activeDriver.waitFor("abort_resolved", event => event.id === "abort-old"),
      ]), "provider abort and delayed real agent_end barrier");
      assert.equal(abortEvent.type, "abort_resolved");
      assert.equal(activeHost.oldEndGateHeld(), true, "the first actual extension agent_end must still be blocked");

      activeDriver.send({ id: "replacement-prompt", type: "prompt", label: "replacement", prompt: REPLACEMENT_PROMPT });
      await activeDriver.waitFor("prompt_called", event => event.id === "replacement-prompt");
      const replacementToolCall = await withDeadline(activeHost.replacementFollowup, "replacement model tool-call response");
      assert.equal(replacementToolCall.body.stream, true);
      const replacementToolResultRequest = await withDeadline(
        activeHost.replacementFinal,
        "provider context containing the replacement tool result",
      );
      assert.equal(
        toolOutputWasSent(replacementToolResultRequest),
        true,
        "the real host must send the completed bash output to its next generation",
      );
      assert.equal(activeHost.oldEndGateHeld(), true, "the old agent_end remains delayed through replacement tool execution");

      activeHost.releaseOldEndGate();
      await activeDriver.waitFor("agent_end_observed", event => event.sequence === 1);
      activeHost.completeReplacement();
      await activeDriver.waitFor("agent_end_observed", event => event.sequence === 2);
      await activeDriver.waitFor("prompt_resolved", event => event.id === "replacement-prompt");

      activeDriver.send({ id: "shutdown", type: "shutdown" });
      shutdownSent = true;
      await activeDriver.waitFor("shutdown_resolved", event => event.id === "shutdown");
      const exit = await withDeadline(activeDriver.waitForExit(), "host SDK driver shutdown", 5_000);
      assert.equal(exit.code, 0, `host SDK driver failed:\\n${activeDriver.stderr()}`);

      assert.ok(activeCapture.requests.length > 0, "the real Langfuse exporter must deliver to the local capture server");
      const spans = activeCapture.spans();
      const roots = spans.filter(span => span.name === "omp-agent");
      assert.equal(roots.length, 2, "the actual old and replacement prompts must each produce one trace root");
      const oldRoot = roots.find(root => promptOf(root) === INTERRUPTED_PROMPT);
      const replacementRoot = roots.find(root => promptOf(root) === REPLACEMENT_PROMPT);
      assert.ok(oldRoot, "the interrupted prompt root must be exported");
      assert.ok(replacementRoot, "the replacement prompt root must be exported");
      assert.equal(
        oldRoot.attrs["langfuse.observation.metadata.cancelled"],
        "true",
        "the interrupted root must be marked cancelled",
      );
      assert.equal(contentOf(replacementRoot.attrs["langfuse.observation.output"]), FINAL_OUTPUT);

      const replacementChildren = spans.filter(
        span => span.traceId === replacementRoot.traceId && span.spanId !== replacementRoot.spanId,
      );
      const generations = replacementChildren.filter(span => span.name === "call-llm");
      const tools = replacementChildren.filter(span => span.name === "bash");
      assert.equal(generations.length, 2, "both replacement generations must remain attached to the replacement trace");
      assert.equal(tools.length, 1, "the replacement bash call must be exported exactly once");
      const tracedTool = tools[0]!;
      assert.equal(tracedTool.parentSpanId, replacementRoot.spanId);
      assert.equal(String(tracedTool.attrs["langfuse.observation.output"]).includes(TOOL_OUTPUT), true);
      assert.ok(generations.some(generation => contentOf(generation.attrs["langfuse.observation.output"]) === FINAL_OUTPUT));
      const latestChildEndNs = replacementChildren.reduce(
        (latest, span) => span.endNs > latest ? span.endNs : latest,
        0n,
      );
      assert.ok(
        replacementRoot.endNs >= latestChildEndNs,
        "the replacement root's own agent_end must close it after its generations and tool result",
      );
      assert.ok(replacementRoot.endNs > replacementRoot.startNs, "the replacement root must have a real closed interval");
      assert.equal(activeHost.requests.length, 3, "only the interrupted stream and replacement generations reach the mock provider");
    } finally {
      if (host) {
        host.releaseOldEndGate();
        try {
          host.completeReplacement();
        } catch {}
      }
      if (driver && driver.child.exitCode === null) {
        if (!shutdownSent && driver.child.stdin?.writable) {
          try {
            driver.send({ id: "cleanup-abort", type: "abort" });
            driver.send({ id: "cleanup-shutdown", type: "shutdown" });
          } catch {}
        }
        try {
          await withDeadline(driver.waitForExit(), "SDK session cleanup", 5_000);
        } catch {
          driver.child.kill("SIGKILL");
          await withDeadline(driver.waitForExit(), "SDK driver termination", 5_000).catch(() => undefined);
        }
      }
      await host?.close().catch(() => undefined);
      capture?.close();
      if (agentDir) rmSync(agentDir, { recursive: true, force: true });
      if (cwd) rmSync(cwd, { recursive: true, force: true });
    }
  });
});
