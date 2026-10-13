import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { cacheAssessmentFor, modelSwitchFor, registerCacheAccounting } from "../src/cache-accounting.ts";

type Handler = (event: unknown, ctx: unknown) => void;
interface Model {
  provider: string;
  id: string;
  api: string;
}
interface TurnOptions {
  advanceMs?: number;
  credentialId?: number;
  messages?: Array<{ role: string; content: string }>;
}

/** Drives the real accounting handlers; only the host boundary and the clock are controlled. */
function fixture(sessionId: string) {
  const handlers = new Map<string, Handler[]>();
  let clock = Date.parse("2026-10-10T05:00:00.000Z");
  const api = {
    on: (name: string, handler: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), handler]),
  } as unknown as ExtensionAPI;
  registerCacheAccounting(api, { now: () => clock });
  let model: Model = { provider: "ollama-cloud", id: "kimi-k3", api: "openai-completions" };
  const emit = (name: string, event: unknown) => {
    for (const handler of handlers.get(name) ?? []) handler(event, { sessionManager: { getSessionId: () => sessionId }, model });
  };
  const history = [
    { role: "system", content: "rules" },
    { role: "user", content: "task" },
  ];
  const reply = (stopReason: string, usage: { input: number; cacheRead: number; cacheWrite: number }, credentialId?: number) => ({
    role: "assistant",
    provider: model.provider,
    model: model.id,
    api: model.api,
    stopReason,
    credentialId,
    content: [],
    usage: { ...usage, output: 10 },
  });
  /** One request and its reply with the provider-reported usage split. */
  const turn = (usage: { input: number; cacheRead: number; cacheWrite?: number }, options: TurnOptions = {}) => {
    clock += options.advanceMs ?? 1_000;
    const payload = { model: model.id, messages: options.messages ?? [...history] };
    emit("before_provider_request", { type: "before_provider_request", payload });
    clock += 500;
    const message = reply("toolUse", { cacheWrite: 0, ...usage }, options.credentialId ?? 1);
    emit("message_end", { type: "message_end", message });
    history.push({ role: "assistant", content: `reply ${history.length}` }, { role: "tool", content: `result ${history.length}` });
    return { assessment: cacheAssessmentFor(message), payload };
  };
  /** A request whose reply was interrupted: it reports no usable usage. */
  const interrupted = () => {
    emit("before_provider_request", { type: "before_provider_request", payload: { model: model.id, messages: [...history] } });
    const message = reply("aborted", { input: 0, cacheRead: 0, cacheWrite: 0 });
    emit("message_end", { type: "message_end", message });
    return message;
  };
  return {
    turn,
    interrupted,
    history,
    setModel: (next: Model) => {
      model = next;
    },
  };
}

describe("cache accounting", () => {
  it("measures each reply against what the previous reply left reusable, not a session-wide share", () => {
    const f = fixture("accounting-progression");
    const first = f.turn({ input: 50_000, cacheRead: 0 }).assessment;
    assert.equal(first?.classification, "first-observed");
    assert.equal(first?.coldStart, true);

    // Implicit caches serve the whole previous prompt except its last token.
    const warm = f.turn({ input: 300, cacheRead: 49_999 }).assessment;
    assert.equal(warm?.classification, "append-only-progression");
    assert.equal(warm?.missedReusable, 0);
    assert.equal(warm?.newTokens, 299);

    // Nothing changed on the wire, yet the cache is gone: the whole prefix is re-processed.
    const dropped = f.turn({ input: 50_400, cacheRead: 0 }).assessment;
    assert.equal(dropped?.classification, "unexplained-loss");
    assert.equal(dropped?.appendOnly, true);
    assert.equal(dropped?.missedReusable, 50_299);
    assert.deepEqual(
      { requests: dropped?.session.requests, losses: dropped?.session.losses, missed: dropped?.session.missedReusable },
      { requests: 3, losses: 1, missed: 50_299 },
    );
  });

  it("attributes losses to idle expiry, an interrupt, an account switch and a prefix rewrite", () => {
    const f = fixture("accounting-causes");
    f.turn({ input: 40_000, cacheRead: 0 });

    const idle = f.turn({ input: 40_300, cacheRead: 0 }, { advanceMs: 6 * 60_000 }).assessment;
    assert.equal(idle?.classification, "ttl-sized-idle");

    const aborted = f.interrupted();
    assert.equal(cacheAssessmentFor(aborted), undefined, "partial usage of an interrupted reply is not assessed");
    const afterInterrupt = f.turn({ input: 20_000, cacheRead: 20_600 }).assessment;
    assert.equal(afterInterrupt?.classification, "after-interrupt");
    assert.equal(afterInterrupt?.disrupted, true);

    const switched = f.turn({ input: 41_000, cacheRead: 0 }, { credentialId: 2 }).assessment;
    assert.equal(switched?.classification, "account-switch");
    assert.deepEqual(switched?.accountSwitch, { provider: "ollama-cloud", from: 1, to: 2 });

    const rewritten = [...f.history];
    rewritten[1] = { role: "user", content: "edited task" };
    const rewrite = f.turn({ input: 41_500, cacheRead: 0 }, { credentialId: 2, messages: rewritten }).assessment;
    assert.equal(rewrite?.classification, "prefix-rewrite");
    assert.equal(rewrite?.appendOnly, false);
  });

  it("labels a switch to another model and resumes the first model against its own cache", () => {
    const f = fixture("accounting-model-switch");
    f.turn({ input: 30_000, cacheRead: 0 });
    f.setModel({ provider: "kimi-code", id: "k3", api: "anthropic-messages" });
    const { assessment: other, payload } = f.turn({ input: 30_400, cacheRead: 0 });
    assert.equal(other?.classification, "model-switch");
    assert.equal(other?.coldStart, true);
    assert.deepEqual(modelSwitchFor(payload), {
      from: { provider: "ollama-cloud", model: "kimi-k3" },
      to: { provider: "kimi-code", model: "k3" },
    });

    f.setModel({ provider: "ollama-cloud", id: "kimi-k3", api: "openai-completions" });
    const back = f.turn({ input: 900, cacheRead: 29_999 }).assessment;
    assert.equal(back?.classification, "append-only-progression");
    assert.deepEqual(back?.modelSwitch?.from, { provider: "kimi-code", model: "k3" });
  });

  it("flags a held but non-advancing cache as a stalled tail", () => {
    const f = fixture("accounting-stalled");
    f.setModel({ provider: "anthropic", id: "claude", api: "anthropic-messages" });
    f.turn({ input: 2_000, cacheRead: 0, cacheWrite: 300_000 });
    const stalled = f.turn({ input: 4_000, cacheRead: 300_000 }).assessment;
    assert.equal(stalled?.classification, "stalled-tail");
    assert.equal(stalled?.missedReusable, 2_000);
  });
});
