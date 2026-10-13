import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

/** Hold the first real agent_end notification before later extensions receive it. */
export default function delayFirstAgentEnd(pi: ExtensionAPI): void {
  let agentEnds = 0;
  pi.on("agent_end", async () => {
    agentEnds++;
    if (agentEnds !== 1) return;

    const gateUrl = process.env.HOST_INTERRUPT_GATE_URL;
    if (!gateUrl) throw new Error("HOST_INTERRUPT_GATE_URL is required by the interruption fixture");
    const response = await fetch(gateUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sequence: agentEnds }),
      signal: AbortSignal.timeout(30_000),
    });
    if (!response.ok) throw new Error(`agent_end gate failed with HTTP ${response.status}`);
    const released: unknown = await response.json();
    if (
      !released ||
      typeof released !== "object" ||
      Array.isArray(released) ||
      !("released" in released) ||
      released.released !== true
    ) {
      throw new Error("agent_end gate returned without an explicit release");
    }
  });
}
