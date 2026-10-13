import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { after, before, describe, it } from "node:test";
import { extractModelParameters } from "../src/index.ts";
import {
  type Capture,
  type CapturedSpan,
  type Sandbox,
  createSandbox,
  runPi,
  startCaptureServer,
  startMockProvider,
  waitForRequests,
} from "./helpers.ts";

describe("extractModelParameters", () => {
  const plain = { reasoning: false };
  const PARAM_CHAR_CAP = 200;

  it("reads the cap the payload really carries, whatever the dialect calls it", () => {
    const payloads: Array<[string, unknown]> = [
      ["anthropic-messages / openai-completions", { max_tokens: 1477 }],
      ["openai-completions on newer OpenAI models", { max_completion_tokens: 1477 }],
      ["openai-responses / azure-openai-responses", { max_output_tokens: 1477 }],
      ["mistral-conversations", { maxTokens: 1477 }],
      ["google-generative-ai / google-vertex", { config: { maxOutputTokens: 1477 } }],
      ["bedrock-converse-stream", { inferenceConfig: { maxTokens: 1477 } }],
      ["pi-messages", { options: { maxTokens: 1477 } }],
    ];
    for (const [dialect, payload] of payloads) {
      assert.deepEqual(extractModelParameters(payload, plain), { max_tokens: 1477 }, dialect);
    }
  });

  it("reports the clamped cap, not the model's configured one", () => {
    assert.deepEqual(extractModelParameters({ max_tokens: 1477 }, { reasoning: false }), { max_tokens: 1477 });
  });

  it("adds the thinking level only for reasoning models", () => {
    assert.deepEqual(extractModelParameters({ max_tokens: 8192 }, { reasoning: true }, "high"), {
      max_tokens: 8192,
      thinking_level: "high",
    });
    assert.deepEqual(extractModelParameters({ max_tokens: 8192 }, plain, "high"), { max_tokens: 8192 });
  });

  it("treats `off` and a missing level as no thinking", () => {
    assert.deepEqual(extractModelParameters({ max_tokens: 4096 }, { reasoning: true }, "off"), { max_tokens: 4096 });
    assert.deepEqual(extractModelParameters({ max_tokens: 4096 }, { reasoning: true }), { max_tokens: 4096 });
  });

  it("returns undefined when nothing is worth a chip", () => {
    assert.equal(extractModelParameters(undefined, plain), undefined);
    assert.equal(extractModelParameters({}, plain), undefined);
    assert.equal(extractModelParameters("not an object", plain), undefined);
    assert.equal(extractModelParameters({ max_tokens: 0 }, plain), undefined);
    assert.equal(extractModelParameters({}, undefined), undefined);
  });

  it("keeps the cap when the context has no model", () => {
    assert.deepEqual(extractModelParameters({ max_tokens: 8192 }, undefined, "high"), { max_tokens: 8192 });
  });

  it("keeps the thinking level when the payload has no cap", () => {
    assert.deepEqual(extractModelParameters({}, { reasoning: true }, "medium"), { thinking_level: "medium" });
  });

  it("reads the thinking budget the provider was really given", () => {
    const reasoning = { reasoning: true };
    const payloads: Array<[string, unknown]> = [
      ["anthropic-messages", { max_tokens: 8192, thinking: { type: "enabled", budget_tokens: 7168 } }],
      [
        "bedrock-converse-stream",
        {
          inferenceConfig: { maxTokens: 8192 },
          additionalModelRequestFields: { thinking: { type: "enabled", budget_tokens: 7168 } },
        },
      ],
      ["google", { config: { maxOutputTokens: 8192, thinkingConfig: { thinkingBudget: 7168 } } }],
      ["openai-completions (vLLM)", { max_tokens: 8192, thinking_token_budget: 7168 }],
      ["openai-completions (thinking_budget)", { max_tokens: 8192, thinking_budget: 7168 }],
      ["openai-completions (thinking_budget_tokens)", { max_tokens: 8192, thinking_budget_tokens: 7168 }],
    ];
    for (const [dialect, payload] of payloads) {
      assert.deepEqual(
        extractModelParameters(payload, reasoning, "high"),
        { max_tokens: 8192, thinking_level: "high", thinking_budget_tokens: 7168 },
        dialect,
      );
    }
  });

  it("omits the budget when the provider gets none", () => {
    assert.deepEqual(
      extractModelParameters({ max_tokens: 8192, thinking: { type: "adaptive" }, output_config: { effort: "high" } }, { reasoning: true }, "high"),
      { max_tokens: 8192, thinking_level: "high" },
    );
    assert.deepEqual(extractModelParameters({ config: { thinkingConfig: { thinkingBudget: 0 } } }, plain), undefined);
  });

  it("reports the prompt cache retention both dialects express differently", () => {
    assert.deepEqual(extractModelParameters({ prompt_cache_retention: "24h" }, plain), {
      prompt_cache_retention: "24h",
    });
    assert.deepEqual(
      extractModelParameters({ max_tokens: 8192, system: [{ type: "text", cache_control: { type: "ephemeral", ttl: "1h" } }] }, plain),
      { max_tokens: 8192, prompt_cache_retention: "1h" },
    );
    assert.deepEqual(
      extractModelParameters({ max_tokens: 8192, system: [{ type: "text", cache_control: { type: "ephemeral" } }] }, plain),
      { max_tokens: 8192 },
    );
  });

  it("reports the service tier and the tool choice when they are set", () => {
    assert.deepEqual(extractModelParameters({ service_tier: "flex", tool_choice: "auto" }, plain), {
      service_tier: "flex",
      tool_choice: "auto",
    });
    assert.deepEqual(extractModelParameters({ tool_choice: { type: "any" } }, plain), { tool_choice: "any" });
    assert.deepEqual(extractModelParameters({ tool_choice: { type: "function", function: { name: "read" } } }, plain), {
      tool_choice: "function",
    });
  });

  it("reports the sampling parameters the model declares, with the values the wire carried", () => {
    const samplingParams = { temperature: 0.2, top_p: 0.9, top_k: 40, min_p: 0.05, repetition_penalty: 1.1 };
    const onTheWire = { temperature: 0.7, top_p: 0.5, top_k: 10, min_p: 0.01, repetition_penalty: 1.9 };
    assert.deepEqual(extractModelParameters({ max_tokens: 8192, ...onTheWire }, { reasoning: false, samplingParams }), {
      max_tokens: 8192,
      ...onTheWire,
    });
  });

  it("serializes a non-scalar sampling parameter instead of dropping it", () => {
    const samplingParams = { stop: ["</done>"], seed: 7 };
    assert.deepEqual(extractModelParameters({ ...samplingParams }, { reasoning: false, samplingParams }), {
      stop: '["</done>"]',
      seed: 7,
    });
  });

  it("never lets a sampling parameter pull a payload structure onto the span", () => {
    const conversation = [{ role: "user", content: "private user text ".repeat(40) }];
    for (const key of ["messages", "system", "tools", "input", "contents"]) {
      const out = extractModelParameters(
        { max_tokens: 8192, [key]: conversation },
        { reasoning: false, samplingParams: { [key]: 1 } },
      );
      assert.deepEqual(out, { max_tokens: 8192 }, key);
    }
    const long = "x".repeat(PARAM_CHAR_CAP + 1);
    assert.deepEqual(extractModelParameters({ note: long }, { reasoning: false, samplingParams: { note: 1 } }), undefined);
    const short = "x".repeat(PARAM_CHAR_CAP);
    assert.deepEqual(extractModelParameters({ note: short }, { reasoning: false, samplingParams: { note: 1 } }), {
      note: short,
    });
  });

  it("keeps the chips it already collected when a late sampling parameter blows up", () => {
    const cyclic: Record<string, unknown> = {};
    cyclic.self = cyclic;
    assert.deepEqual(
      extractModelParameters({ max_tokens: 1477, weird: cyclic }, { reasoning: true, samplingParams: { weird: 1 } }, "high"),
      { max_tokens: 1477, thinking_level: "high" },
    );
  });

  it("continues collecting later parameters after a cyclic or BigInt sampling value", () => {
    const cyclic: Record<string, unknown> = {};
    cyclic.self = cyclic;
    const samplingParams = { broken: 1, integer: 1, valid_after: 1 };
    assert.deepEqual(extractModelParameters({
      broken: cyclic, integer: 1n, valid_after: "retained",
    }, { samplingParams }), { valid_after: "retained" });
  });

  it("stops traversing an oversized parameter before reaching an expensive trailing getter", () => {
    let reads = 0;
    const oversized = {
      first: "x".repeat(PARAM_CHAR_CAP + 1),
      get last() { reads++; return "must not be visited"; },
    };
    assert.deepEqual(extractModelParameters({
      oversized, valid_after: 7,
    }, { samplingParams: { oversized: 1, valid_after: 1 } }), { valid_after: 7 });
    assert.equal(reads, 0);
  });

  it("rejects a cap that is not a positive whole number", () => {
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -5, 1.5, "8192", null, true]) {
      assert.equal(extractModelParameters({ max_tokens: bad }, plain), undefined, String(bad));
    }
  });

  it("never throws on a hostile payload", () => {
    const hostile: Record<string, unknown> = {};
    Object.defineProperty(hostile, "max_tokens", {
      enumerable: true,
      get() {
        throw new Error("payload getter exploded");
      },
    });
    assert.equal(extractModelParameters(hostile, plain), undefined);
  });

  it("omits a declared sampling parameter the dialect never sent", () => {
    assert.deepEqual(
      extractModelParameters({ max_tokens: 8192 }, { reasoning: false, samplingParams: { temperature: 0.2 } }),
      { max_tokens: 8192 },
    );
  });

  it("never descends into message arrays looking for a cap", () => {
    const payload = {
      messages: [{ role: "user", content: "budget_tokens: 99, max_tokens: 99" }, { max_tokens: 99 }],
      max_tokens: 1477,
    };
    assert.deepEqual(extractModelParameters(payload, plain), { max_tokens: 1477 });
    assert.deepEqual(extractModelParameters({ messages: [{ max_tokens: 99 }] }, plain), undefined);
  });
});

function langfuseEnv(capture: Capture): Record<string, string> {
  return {
    LANGFUSE_PUBLIC_KEY: "pk-lf-test",
    LANGFUSE_SECRET_KEY: "sk-lf-test",
    LANGFUSE_BASE_URL: `http://127.0.0.1:${capture.port}`,
  };
}

function generationParameters(spans: CapturedSpan[]): Array<Record<string, unknown> | undefined> {
  return spans
    .filter((s) => s.name === "call-llm")
    .sort((a, b) => (a.startNs < b.startNs ? -1 : 1))
    .map((s) => {
      const raw = s.attrs["langfuse.observation.model.parameters"];
      return raw === undefined ? undefined : (JSON.parse(String(raw)) as Record<string, unknown>);
    });
}

function enableReasoning(sandbox: Sandbox, level: string) {
  const modelsPath = join(sandbox.agentDir, "models.json");
  const models = JSON.parse(readFileSync(modelsPath, "utf8"));
  models.providers.mock.models[0].reasoning = true;
  writeFileSync(modelsPath, JSON.stringify(models));
  writeFileSync(join(sandbox.agentDir, "settings.json"), JSON.stringify({ defaultThinkingLevel: level }));
}

describe("integration: model parameters", () => {
  let mock: { port: number; close: () => void };

  before(async () => {
    mock = await startMockProvider();
  });
  after(() => mock.close());

  it("records max tokens on every generation of a plain model", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Explore this project and summarize it", { env: langfuseEnv(capture) });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const params = generationParameters(capture.spans());
      assert.equal(params.length, 3, "the mock script runs three generations");
      for (const p of params) {
        assert.deepEqual(p, { max_tokens: 8192 });
      }
    } finally {
      capture.close();
    }
  });

  it("adds the thinking level for a reasoning model", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      enableReasoning(sandbox, "high");
      const result = await runPi(sandbox, "Explore this project and summarize it", { env: langfuseEnv(capture) });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const params = generationParameters(capture.spans());
      assert.equal(params.length, 3);
      for (const p of params) {
        assert.deepEqual(p, { max_tokens: 8192, thinking_level: "high", reasoning_effort: "high" });
      }
    } finally {
      capture.close();
    }
  });

  it("reports the cap pi clamped to the context, not the model's 8192", async () => {
    const capture = await startCaptureServer();
    const wired = await startMockProvider();
    try {
      const sandbox = createSandbox(wired.port, { contextWindow: 4000, readmeFillerLines: 700 });
      writeFileSync(join(sandbox.agentDir, "config.yml"), JSON.stringify({ compaction: { enabled: false } }));
      const result = await runPi(sandbox, "Explore this project and summarize it", { env: langfuseEnv(capture) });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const caps = generationParameters(capture.spans()).map((p) => p?.max_tokens as number);
      assert.equal(caps.length, 3);
      const sentCaps = wired.payloads().map(payload => payload.max_tokens ?? payload.max_completion_tokens);
      assert.deepEqual(caps, sentCaps, "each generation must report the cap its provider request actually carried");
      assert.ok(caps.some(cap => cap > 0 && cap < 8192), "the fixture must exercise a real context-clamped request");
    } finally {
      wired.close();
      capture.close();
    }
  });

  it("does not invent a Responses-only retention parameter on Chat Completions", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const modelsPath = join(sandbox.agentDir, "models.json");
      const models = JSON.parse(readFileSync(modelsPath, "utf8"));
      models.providers.mock.models[0].compat = { supportsLongCacheRetention: true };
      writeFileSync(modelsPath, JSON.stringify(models));
      const result = await runPi(sandbox, "Explore this project and summarize it", {
        env: { ...langfuseEnv(capture), PI_CACHE_RETENTION: "long" },
      });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const params = generationParameters(capture.spans());
      assert.equal(params.length, 3);
      for (const p of params) {
        assert.equal(p?.prompt_cache_retention, undefined);
      }
    } finally {
      capture.close();
    }
  });

  it("reports exactly the sampling parameters the request carried", async () => {
    const capture = await startCaptureServer();
    const wired = await startMockProvider();
    try {
      const sandbox = createSandbox(wired.port);
      const modelsPath = join(sandbox.agentDir, "models.json");
      const models = JSON.parse(readFileSync(modelsPath, "utf8"));
      const declared: Record<string, unknown> = { temperature: 0.2, top_k: 40, stop: ["</done>"] };
      models.providers.mock.models[0].samplingParams = declared;
      writeFileSync(modelsPath, JSON.stringify(models));
      const result = await runPi(sandbox, "Explore this project and summarize it", { env: langfuseEnv(capture) });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const sent = wired.payloads();
      const params = generationParameters(capture.spans());
      assert.equal(params.length, sent.length, "one generation per provider request");
      const keys = Object.keys(declared);
      for (const [i, p] of params.entries()) {
        assert.equal(p?.max_tokens, 8192, `generation ${i + 1} must still carry the cap`);
        assert.deepEqual(
          keys.filter((k) => p?.[k] !== undefined),
          keys.filter((k) => sent[i]?.[k] !== undefined),
          `generation ${i + 1} must report exactly the declared parameters the wire carried`,
        );
        assert.equal(p?.temperature, sent[i]?.temperature, `generation ${i + 1} temperature`);
        assert.equal(p?.top_k, sent[i]?.top_k, `generation ${i + 1} top_k`);
      }
    } finally {
      wired.close();
      capture.close();
    }
  });

  it("puts the context window on every generation", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port, { contextWindow: 18000 });
      const result = await runPi(sandbox, "Explore this project and summarize it", { env: langfuseEnv(capture) });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const generations = capture.spans().filter((s) => s.name === "call-llm");
      assert.equal(generations.length, 3);
      for (const g of generations) {
        assert.equal(g.attrs["langfuse.observation.metadata.provider"], "mock");
        assert.equal(Number(g.attrs["langfuse.observation.metadata.context_window"]), 18000);
      }
    } finally {
      capture.close();
    }
  });
});
