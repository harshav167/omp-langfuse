import { z } from "@oh-my-pi/pi-coding-agent";
import type { ToolDefinitionInput, TracingHistoryMessage } from "../index.ts";

export const recordSchema = z.record(z.string(), z.unknown());
export const arraySchema = z.array(z.unknown());
export const historyPartSchema = z.union([
  z.object({ type: z.literal("text"), text: z.string() }).passthrough(),
  z.object({ type: z.literal("image"), data: z.string(), mimeType: z.string() }).passthrough(),
  z.object({ type: z.literal("thinking"), thinking: z.string() }).passthrough(),
  z.object({ type: z.literal("redactedThinking"), data: z.string() }).passthrough(),
  z.object({ type: z.literal("toolCall"), id: z.string(), name: z.string(), arguments: z.unknown() }).passthrough(),
]);
export const historyMessageSchema = z.object({
  role: z.enum(["user", "assistant", "toolResult"]),
  content: z.union([z.string(), z.array(historyPartSchema)]),
  toolCallId: z.string().optional(),
  toolName: z.string().optional(),
  isError: z.boolean().optional(),
});
export const normalizedRequestSchema = z.object({
  systemPrompt: z.array(z.string()),
  messages: z.array(historyMessageSchema),
  tools: z.array(z.object({ name: z.string(), description: z.string().optional(), parameters: z.unknown().optional() })),
  delta: z.boolean(),
  previousResponseId: z.string().optional(),
  warnings: z.array(z.string()),
  inferredToolCallIds: z.array(z.string()).optional(),
});

export function readRecord(value: unknown): Record<string, unknown> | undefined {
  const parsed = recordSchema.safeParse(value);
  return parsed.success ? parsed.data : undefined;
}

export function readArray(value: unknown): unknown[] | undefined {
  const parsed = arraySchema.safeParse(value);
  return parsed.success ? parsed.data : undefined;
}

/** A data adapter's output, consumed only by the existing tracing handlers. */
export interface NormalizedRequest {
  systemPrompt: string[];
  messages: TracingHistoryMessage[];
  tools: ToolDefinitionInput[];
  delta: boolean;
  previousResponseId?: string;
  warnings: string[];
  /** Positional Google history identities, not IDs observed in the provider payload. */
  inferredToolCallIds?: string[];
}
