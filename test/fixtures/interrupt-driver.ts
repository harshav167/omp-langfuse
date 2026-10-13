import { Database } from "bun:sqlite";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import {
  AuthStorage,
  ModelRegistry,
  SessionManager,
  Settings,
  SqliteAuthCredentialStore,
  USER_INTERRUPT_LABEL,
  createAgentSession,
  type ExtensionFactory,
  type AgentSession,
} from "@oh-my-pi/pi-coding-agent";
import { initializeExtensions } from "@oh-my-pi/pi-coding-agent/modes/runtime-init";
import { createInterface } from "node:readline";
import { join, resolve } from "node:path";

interface DriverCommand {
  id: string;
  type: "prompt" | "abort" | "shutdown";
  label?: string;
  prompt?: string;
}
function parseDriverCommand(value: unknown): DriverCommand {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    !("id" in value) ||
    typeof value.id !== "string" ||
    !("type" in value) ||
    (value.type !== "prompt" && value.type !== "abort" && value.type !== "shutdown")
  ) {
    throw new Error("Invalid host SDK driver command");
  }
  const command: DriverCommand = { id: value.id, type: value.type };
  if ("label" in value) {
    if (typeof value.label !== "string") throw new Error("Driver command label must be a string");
    command.label = value.label;
  }
  if ("prompt" in value) {
    if (typeof value.prompt !== "string") throw new Error("Driver command prompt must be a string");
    command.prompt = value.prompt;
  }
  return command;
}


function writeEvent(event: Record<string, unknown>): Promise<void> {
  const { promise, resolve, reject } = Promise.withResolvers<void>();
  process.stdout.write(`${JSON.stringify(event)}\n`, error => {
    if (error) reject(error);
    else resolve();
  });
  return promise;
}

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required by the host-interruption driver`);
  return value;
}

async function runDriver(): Promise<void> {
  const cwd = requiredEnv("HOST_INTERRUPT_CWD");
  const agentDir = requiredEnv("HOST_INTERRUPT_AGENT_DIR");
  const providerPort = Number(requiredEnv("HOST_INTERRUPT_PROVIDER_PORT"));
  if (!Number.isInteger(providerPort) || providerPort < 1) throw new Error("Invalid mock provider port");

  const fixturePath = resolve(import.meta.dirname, "delayed-agent-end.ts");
  const tracingExtensionPath = resolve(import.meta.dirname, "../../src/index.ts");
  const gateObserver: ExtensionFactory = pi => {
    let observedEnds = 0;
    pi.on("agent_end", async () => {
      observedEnds++;
      await writeEvent({ type: "agent_end_observed", sequence: observedEnds });
    });
  };

  const settings = await Settings.init({ cwd, agentDir });
  const authStorage = new AuthStorage(new SqliteAuthCredentialStore(new Database(":memory:")));
  authStorage.keys.setRuntime("openai", "test-only-local-provider-key");
  const modelRegistry = new ModelRegistry(authStorage, join(agentDir, "models.yml"), { settings });
  const model = buildModel({
    id: "host-interruption-mock",
    name: "Host Interruption Mock",
    api: "openai-completions",
    provider: "openai",
    baseUrl: `http://127.0.0.1:${providerPort}/v1`,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32_000,
    maxTokens: 1_024,
  });

  let session: AgentSession | undefined;
  const prompts: Promise<unknown>[] = [];
  try {
    const created = await createAgentSession({
      cwd,
      agentDir,
      sessionManager: SessionManager.inMemory(cwd),
      settings,
      modelRegistry,
      authStorage,
      model,
      getApiKey: async () => "test-only-local-provider-key",
      disableExtensionDiscovery: true,
      additionalExtensionPaths: [fixturePath, tracingExtensionPath],
      extensions: [gateObserver],
      preloadedCustomToolPaths: [],
      skills: [],
      rules: [],
      contextFiles: [],
      slashCommands: [],
      promptTemplates: [],
      enableMCP: false,
      enableLsp: false,
      enableIrc: false,
      skipPythonPreflight: true,
      toolNames: ["bash"],
      cacheWarming: false,
      autoApprove: true,
    });
    const activeSession = created.session;
    session = activeSession;
    const loadedExtensions = created.extensionsResult.extensions.map(extension => extension.resolvedPath);
    const fixtureIndex = loadedExtensions.indexOf(fixturePath);
    const tracingIndex = loadedExtensions.indexOf(tracingExtensionPath);
    const observerIndex = loadedExtensions.findIndex(path => path.startsWith("<inline-"));
    if (fixtureIndex < 0 || tracingIndex <= fixtureIndex || observerIndex <= tracingIndex) {
      throw new Error(`Required extension order did not load: ${JSON.stringify(loadedExtensions)}`);
    }
    const extensionOrder = loadedExtensions.map((path, index) => {
      if (path === fixturePath) return "delay-first-agent_end";
      if (path === tracingExtensionPath) return "langfuse-tracing";
      if (index === observerIndex) return "post-tracing-agent_end-observer";
      if (path.startsWith("<inline-")) return "sdk-internal-inline-extension";
      return "other-explicit-extension";
    });

    await initializeExtensions(activeSession, {
      mode: "print",
      reportSendError: (action, error) => {
        void writeEvent({ type: "extension_send_error", action, error: error.message });
      },
      reportRuntimeError: error => {
        void writeEvent({ type: "extension_runtime_error", error: String(error) });
      },
    });
    let firstProviderTextDeltaObserved = false;
    activeSession.subscribe(event => {
      if (
        firstProviderTextDeltaObserved ||
        event.type !== "message_update" ||
        event.assistantMessageEvent.type !== "text_delta" ||
        !event.assistantMessageEvent.delta
      ) {
        return;
      }
      firstProviderTextDeltaObserved = true;
      void writeEvent({ type: "provider_text_delta_observed" });
    });


    await writeEvent({
      type: "ready",
      target: "installed OMP createAgentSession + AgentSession",
      options: {
        cwd,
        agentDir,
        sessionManager: "in-memory",
        provider: "local OpenAI-compatible mock",
        providerBaseUrl: model.baseUrl,
        extensions: ["delayed-agent-end.ts", "src/index.ts", "inline agent_end observer"],
        extensionOrder,
        disableExtensionDiscovery: true,
        enableMCP: false,
        enableLsp: false,
        skills: [],
        rules: [],
        toolNames: ["bash"],
        rawRequestsDir: process.env.OMP_RAW_REQUESTS_DIR,
      },
      capabilities: ["session.prompt", "session.abort", "real extension lifecycle events", "real tool execution"],
    });

    const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
    for await (const line of input) {
      let command: DriverCommand;
      try {
        const parsed: unknown = JSON.parse(line);
        command = parseDriverCommand(parsed);
      } catch (error) {
        await writeEvent({ type: "command_error", error: String(error) });
        continue;
      }
      try {
        if (command.type === "prompt") {
          if (!command.prompt) throw new Error("prompt command requires prompt text");
          const promptTask = activeSession.prompt(command.prompt, { throwOnDrop: true });
          prompts.push(promptTask);
          void promptTask.then(
            forwarded => writeEvent({ type: "prompt_resolved", id: command.id, label: command.label, forwarded }),
            error => writeEvent({ type: "prompt_rejected", id: command.id, label: command.label, error: String(error) }),
          );
          await writeEvent({ type: "prompt_called", id: command.id, label: command.label });
        } else if (command.type === "abort") {
          await activeSession.abort({ reason: USER_INTERRUPT_LABEL });
          await writeEvent({ type: "abort_resolved", id: command.id });
        } else if (command.type === "shutdown") {
          await Promise.allSettled(prompts);
          await activeSession.dispose();
          await writeEvent({ type: "shutdown_resolved", id: command.id });
          input.close();
          break;
        } else {
          throw new Error(`Unknown command type: ${String(command.type)}`);
        }
      } catch (error) {
        await writeEvent({ type: "command_error", id: command.id, error: String(error) });
      }
    }
  } finally {
    if (session && !session.isDisposed) await session.dispose().catch(() => undefined);
    authStorage.close();
  }
}

if (import.meta.main) {
  runDriver().catch(error => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
