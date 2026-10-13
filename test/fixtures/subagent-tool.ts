/**
 * Test subagent tool. It starts a child pi process that gets process.env from
 * the parent, the same as the pi subagent example and the gallery packages.
 */
import { spawn } from "node:child_process";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

export default function(pi: ExtensionAPI) {
 const parameters = pi.zod.object({ task: pi.zod.string().describe("Task for the subagent") });
 pi.registerTool({
  name: "subagent",
  label: "Subagent",
  description: "Delegate a task to an isolated child agent.",
  parameters,
  loadMode: "essential",
  async execute(_toolCallId, params) {
   const input = parameters.parse(params);
   const piBin = process.env.TEST_PI_BIN!;
   const childArgs = [
    "--no-extensions", "--no-skills", "--no-rules", "--no-lsp", "--no-title", "--no-prewalk", "--no-pty",
    "-e", process.env.TEST_LANGFUSE_EXTENSION!,
    "--provider", "mock",
    "--model", "mock-gpt-1",
    "--api-key", "mock-key",
    "-p", "--no-session",
    `Task: ${input.task}`,
   ];
   const stdout = await new Promise<string>((resolve, reject) => {
    const child = spawn(piBin, childArgs, { cwd: process.cwd(), shell: false, stdio: ["ignore", "pipe", "pipe"] });
    let buf = "";
    let errors = "";
    child.stdout.on("data", (d: Buffer) => (buf += d.toString()));
    child.stderr.on("data", (d: Buffer) => (errors += d.toString()));
    child.on("error", reject);
    child.on("close", code => code === 0 ? resolve(buf) : reject(new Error(`Subagent exited ${code}: ${errors}`)));
   });

   if (!stdout.trim()) throw new Error("Subagent completed without an answer");
   return { content: [{ type: "text", text: stdout.trim() }], details: {} };
  },
 });
}
