import { z } from "@oh-my-pi/pi-coding-agent";
import type { ToolDefinitionInput, TracingHistoryMessage, TracingHistoryPart } from "../index.ts";
import { arraySchema, normalizedRequestSchema, readArray as readArrayValue, readRecord, recordSchema, type NormalizedRequest } from "./types.ts";
const stringSchema = z.string();
const responsesInputSchema = z.union([stringSchema, arraySchema]);


function warning(warnings: string[], location: string, message: string): void {
  warnings.push(`${location}: ${message}`);
}

function parseRecord(value: unknown, location: string, warnings: string[]): Record<string, unknown> | undefined {
  const parsed = readRecord(value);
  if (parsed) return parsed;
  warning(warnings, location, "expected an object; content was not replayed");
  return undefined;
}

function readString(value: unknown): string | undefined {
  const parsed = stringSchema.safeParse(value);
  return parsed.success ? parsed.data : undefined;
}

function readArray(value: unknown, location: string, warnings: string[]): unknown[] | undefined {
  if (value === undefined) return undefined;
  const parsed = readArrayValue(value);
  if (parsed) return parsed;
  warning(warnings, location, "expected an array");
  return undefined;
}

function parseRoot(payload: unknown, api: string): Record<string, unknown> {
  const parsed = recordSchema.safeParse(payload);
  if (!parsed.success) throw new Error(`Invalid ${api} request: expected a JSON object`);
  return parsed.data;
}

function parseArguments(value: unknown, location: string, warnings: string[]): unknown {
  if (typeof value !== "string") {
    if (value === undefined) warning(warnings, location, "tool arguments were not supplied");
    return value;
  }
  try {
    const parsed: unknown = JSON.parse(value);
    return parsed;
  } catch {
    warning(warnings, location, "tool arguments were not valid JSON; preserving the original string");
    return value;
  }
}

function imageFromDataUri(value: string, location: string, warnings: string[]): TracingHistoryPart | undefined {
  const match = /^data:(image\/[a-z0-9.+-]+);base64,([a-z0-9+/=\s]+)$/i.exec(value);
  if (!match) {
    warning(warnings, location, "image reference is not an embedded base64 data URI and cannot be replayed without IO");
    return undefined;
  }
  const data = match[2]!.replace(/\s+/g, "");
  if (!data || !/^[A-Za-z0-9+/]+={0,2}$/.test(data)) {
    warning(warnings, location, "image data URI has invalid base64 content");
    return undefined;
  }
  return { type: "image", mimeType: match[1]!, data };
}

function imagePart(record: Record<string, unknown>, location: string, warnings: string[]): TracingHistoryPart | undefined {
  const imageUrl = record.image_url;
  if (typeof imageUrl === "string") return imageFromDataUri(imageUrl, location, warnings);
  if (imageUrl !== undefined) {
    const imageRecord = parseRecord(imageUrl, `${location}.image_url`, warnings);
    const url = imageRecord ? readString(imageRecord.url) : undefined;
    if (url !== undefined) return imageFromDataUri(url, location, warnings);
  }
  if (typeof record.file_id === "string" && record.file_id.length > 0) {
    warning(warnings, location, "image is referenced by file_id and cannot be recovered without IO");
  } else {
    warning(warnings, location, "image part did not contain a replayable image_url");
  }
  return undefined;
}

function textFromPart(record: Record<string, unknown>, field: string, location: string, warnings: string[]): string | undefined {
  const text = readString(record[field]);
  if (text === undefined) warning(warnings, location, `expected string field ${field}`);
  return text;
}

function parseContentPart(
  value: unknown,
  location: string,
  warnings: string[],
): TracingHistoryPart | undefined {
  const part = parseRecord(value, location, warnings);
  if (!part) return undefined;
  const type = readString(part.type);
  if (!type) {
    warning(warnings, location, "content part has no string type");
    return undefined;
  }

  switch (type) {
    case "text":
    case "input_text":
    case "output_text": {
      const text = textFromPart(part, "text", location, warnings);
      if (text === undefined) return undefined;
      if (Array.isArray(part.annotations) && part.annotations.length > 0) {
        warning(warnings, location, "text annotations were not represented; the text itself was retained");
      }
      return { type: "text", text };
    }
    case "reasoning_text":
    case "summary_text": {
      const text = textFromPart(part, "text", location, warnings);
      return text === undefined ? undefined : { type: "thinking", thinking: text };
    }
    case "thinking": {
      if (typeof part.thinking === "string") return { type: "thinking", thinking: part.thinking };
      if (part.thinking === undefined) {
        warning(warnings, location, "thinking block has no thinking content");
        return undefined;
      }
      const thoughts = readArray(part.thinking, `${location}.thinking`, warnings);
      if (!thoughts) return undefined;
      const text = thoughts.flatMap((thought, index) => {
        const thoughtRecord = parseRecord(thought, `${location}.thinking[${index}]`, warnings);
        const partTextValue = thoughtRecord ? readString(thoughtRecord.text) : undefined;
        if (partTextValue === undefined && thoughtRecord) {
          warning(warnings, `${location}.thinking[${index}]`, "thinking text was not a string");
        }
        return partTextValue === undefined ? [] : [partTextValue];
      }).join("");
      return { type: "thinking", thinking: text };
    }
    case "redacted_thinking": {
      const data = textFromPart(part, "data", location, warnings);
      return data === undefined ? undefined : { type: "redactedThinking", data };
    }
    case "image_url":
    case "input_image":
    case "output_image":
    case "image":
      return imagePart(part, location, warnings);
    case "refusal": {
      const refusal = readString(part.refusal) ?? readString(part.text);
      if (refusal === undefined) {
        warning(warnings, location, "refusal text was not supplied");
        return undefined;
      }
      warning(warnings, location, "refusal classification is not represented; refusal text was retained");
      return { type: "text", text: refusal };
    }
    case "input_file":
    case "file":
      warning(warnings, location, "file content is not representable in TracingHistoryPart");
      return undefined;
    default:
      warning(warnings, location, `unsupported content part type ${JSON.stringify(type)}`);
      return undefined;
  }
}

function normalizeContent(value: unknown, location: string, warnings: string[]): string | TracingHistoryPart[] {
  if (typeof value === "string") return value;
  if (value === undefined || value === null) return [];
  const items = readArray(value, location, warnings);
  if (!items) return [];
  const parts: TracingHistoryPart[] = [];
  for (let index = 0; index < items.length; index += 1) {
    const part = parseContentPart(items[index], `${location}[${index}]`, warnings);
    if (part) parts.push(part);
  }
  return parts;
}

function contentAsText(content: string | TracingHistoryPart[]): string {
  if (typeof content === "string") return content;
  return content.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("");
}

function addSystemPrompt(
  systemPrompt: string[],
  value: unknown,
  location: string,
  warnings: string[],
): void {
  const content = normalizeContent(value, location, warnings);
  const unsupported = typeof content === "string" ? false : content.some((part) => part.type !== "text");
  if (unsupported) warning(warnings, location, "non-text system/developer content is not representable in systemPrompt");
  const text = contentAsText(content);
  if (text.length > 0) systemPrompt.push(text);
}

function asParts(content: string | TracingHistoryPart[]): TracingHistoryPart[] {
  return typeof content === "string" ? (content ? [{ type: "text", text: content }] : []) : content;
}

function parseChatToolCall(
  value: unknown,
  index: number,
  warnings: string[],
): TracingHistoryPart | undefined {
  const location = `messages[].tool_calls[${index}]`;
  const call = parseRecord(value, location, warnings);
  if (!call) return undefined;
  const id = readString(call.id);
  if (!id) {
    warning(warnings, location, "tool call has no wire id and cannot be represented");
    return undefined;
  }
  const callType = readString(call.type);
  if (callType === "custom" || call.custom !== undefined) {
    const custom = parseRecord(call.custom, `${location}.custom`, warnings);
    const name = custom ? readString(custom.name) : undefined;
    if (!name) {
      warning(warnings, location, "custom tool call has no name and cannot be represented");
      return undefined;
    }
    warning(warnings, location, "custom tool call was represented as a canonical toolCall");
    if (custom?.input === undefined) warning(warnings, location, "custom tool input was not supplied");
    return { type: "toolCall", id, name, arguments: custom?.input };
  }
  if (callType !== undefined && callType !== "function") {
    warning(warnings, location, `unsupported tool call type ${JSON.stringify(callType)}`);
    return undefined;
  }
  const fn = parseRecord(call.function, `${location}.function`, warnings);
  const name = fn ? readString(fn.name) : undefined;
  if (!name) {
    warning(warnings, location, "function tool call has no name and cannot be represented");
    return undefined;
  }
  const args = parseArguments(fn?.arguments, `${location}.function.arguments`, warnings);
  return { type: "toolCall", id, name, arguments: args };
}

function parseLegacyChatFunctionCall(
  value: unknown,
  warnings: string[],
): TracingHistoryPart | undefined {
  const location = "messages[].function_call";
  const call = parseRecord(value, location, warnings);
  if (!call) return undefined;
  const id = readString(call.id);
  const name = readString(call.name);
  if (!id || !name) {
    warning(warnings, location, "legacy function_call has no call id or name; no id was fabricated");
    return undefined;
  }
  return { type: "toolCall", id, name, arguments: parseArguments(call.arguments, `${location}.arguments`, warnings) };
}

function parseChatTools(value: unknown, warnings: string[]): ToolDefinitionInput[] {
  if (value === undefined) return [];
  const entries = readArray(value, "tools", warnings);
  if (!entries) return [];
  const tools: ToolDefinitionInput[] = [];
  for (let index = 0; index < entries.length; index += 1) {
    const location = `tools[${index}]`;
    const item = parseRecord(entries[index], location, warnings);
    if (!item) continue;
    const type = readString(item.type);
    if (type !== undefined && type !== "function") {
      warning(warnings, location, `non-function tool type ${JSON.stringify(type)} is not representable`);
      continue;
    }
    const fn = parseRecord(item.function, `${location}.function`, warnings);
    const name = fn ? readString(fn.name) : undefined;
    if (!name || !name.trim()) {
      warning(warnings, location, "function definition has no name; it was not added as a stub");
      continue;
    }
    const description = fn ? readString(fn.description) : undefined;
    if (fn?.description !== undefined && description === undefined) {
      warning(warnings, location, "function description was not a string");
    }
    tools.push({
      name,
      ...(description !== undefined ? { description } : {}),
      ...(fn && Object.hasOwn(fn, "parameters") ? { parameters: fn.parameters } : {}),
    });
  }
  return tools;
}

function normalizeChatCompletions(root: Record<string, unknown>): NormalizedRequest {
  const warnings: string[] = [];
  const systemPrompt: string[] = [];
  const messages: TracingHistoryMessage[] = [];
  const rawMessages = readArray(root.messages, "messages", warnings);
  if (!rawMessages) throw new Error("Invalid openai-completions request: expected a messages array");

  let sawConversationMessage = false;
  for (let index = 0; index < rawMessages.length; index += 1) {
    const location = `messages[${index}]`;
    const message = parseRecord(rawMessages[index], location, warnings);
    if (!message) continue;
    const role = readString(message.role);
    if (!role) {
      warning(warnings, location, "message has no string role");
      continue;
    }
    if (role === "system" || role === "developer") {
      if (sawConversationMessage) warning(warnings, location, `${role} prompt occurred after conversation content; canonical systemPrompt moves it to the front`);
      if (message.content === undefined || message.content === null) {
        warning(warnings, location, `${role} message has no content`);
        continue;
      }
      addSystemPrompt(systemPrompt, message.content, `${location}.content`, warnings);
      continue;
    }

    if (role === "user" || role === "assistant") {
      sawConversationMessage = true;
      if (role === "user" && (message.content === undefined || message.content === null)) {
        warning(warnings, location, "user message has no content");
        continue;
      }
      let content = normalizeContent(message.content, `${location}.content`, warnings);
      if (role === "assistant") {
        if (message.audio !== undefined && message.audio !== null) {
          warning(warnings, `${location}.audio`, "audio content is not representable in TracingHistoryPart");
        }
        const assistantParts: TracingHistoryPart[] = [];
        for (const field of ["reasoning_content", "reasoning", "reasoning_text"]) {
          const thought = readString(message[field]);
          if (thought !== undefined && thought.length > 0) assistantParts.push({ type: "thinking", thinking: thought });
          else if (message[field] !== undefined && thought === undefined) warning(warnings, `${location}.${field}`, "reasoning field was not a string");
        }
        const thinkingBlocks = readArray(message.thinking_blocks, `${location}.thinking_blocks`, warnings);
        if (thinkingBlocks) {
          for (let blockIndex = 0; blockIndex < thinkingBlocks.length; blockIndex += 1) {
            const blockLocation = `${location}.thinking_blocks[${blockIndex}]`;
            const block = parseRecord(thinkingBlocks[blockIndex], blockLocation, warnings);
            if (!block) continue;
            if (block.type === "thinking" && typeof block.thinking === "string") {
              assistantParts.push({ type: "thinking", thinking: block.thinking });
              if (typeof block.signature === "string") warning(warnings, blockLocation, "thinking signature is not represented in TracingHistoryPart");
            } else if (block.type === "redacted_thinking" && typeof block.data === "string") {
              assistantParts.push({ type: "redactedThinking", data: block.data });
            } else {
              warning(warnings, blockLocation, "unsupported thinking block shape");
            }
          }
        }
        const reasoningDetails = readArray(message.reasoning_details, `${location}.reasoning_details`, warnings);
        if (reasoningDetails) {
          for (let detailIndex = 0; detailIndex < reasoningDetails.length; detailIndex += 1) {
            const detailLocation = `${location}.reasoning_details[${detailIndex}]`;
            const detail = parseRecord(reasoningDetails[detailIndex], detailLocation, warnings);
            if (!detail) continue;
            if (detail.type === "reasoning.text" && typeof detail.text === "string") {
              assistantParts.push({ type: "thinking", thinking: detail.text });
            } else if (detail.type === "reasoning.encrypted" && typeof detail.data === "string") {
              assistantParts.push({ type: "redactedThinking", data: detail.data });
              warning(warnings, detailLocation, "reasoning detail id/signature metadata is not represented");
            } else {
              warning(warnings, detailLocation, "unsupported reasoning detail shape");
            }
          }
        }
        assistantParts.push(...asParts(content));
        const toolCalls = readArray(message.tool_calls, `${location}.tool_calls`, warnings);
        if (toolCalls) {
          for (let callIndex = 0; callIndex < toolCalls.length; callIndex += 1) {
            const call = parseChatToolCall(toolCalls[callIndex], callIndex, warnings);
            if (call) assistantParts.push(call);
          }
        }
        if (message.function_call !== undefined) {
          const legacyCall = parseLegacyChatFunctionCall(message.function_call, warnings);
          if (legacyCall) assistantParts.push(legacyCall);
        }
        if (typeof message.refusal === "string" && message.refusal.length > 0) {
          warning(warnings, `${location}.refusal`, "refusal classification is not represented; refusal text was retained");
          assistantParts.push({ type: "text", text: message.refusal });
        }
        content = assistantParts;
        if (!assistantParts.length) continue;
      }
      messages.push({ role, content });
      continue;
    }

    if (role === "tool" || role === "function") {
      sawConversationMessage = true;
      if (message.content === undefined || message.content === null) {
        warning(warnings, location, "tool result has no content; it was not replayed");
        continue;
      }
      const resultContent = normalizeContent(message.content, `${location}.content`, warnings);
      const toolCallId = readString(message.tool_call_id);
      const toolName = readString(message.name);
      if (!toolCallId && role === "tool") warning(warnings, location, "tool result has no tool_call_id; no id was fabricated");
      if (!toolCallId && role === "function") warning(warnings, location, "legacy function result has no call id");
      messages.push({
        role: "toolResult",
        content: resultContent,
        ...(toolCallId !== undefined ? { toolCallId } : {}),
        ...(toolName !== undefined ? { toolName } : {}),
        ...(message.is_error === true ? { isError: true } : {}),
      });
      continue;
    }

    warning(warnings, location, `unsupported Chat Completions role ${JSON.stringify(role)}`);
  }

  const result: NormalizedRequest = {
    systemPrompt,
    messages,
    tools: parseChatTools(root.tools, warnings),
    delta: false,
    warnings,
  };
  return normalizedRequestSchema.parse(result);
}

function parseResponsesTools(value: unknown, warnings: string[]): ToolDefinitionInput[] {
  if (value === undefined) return [];
  const entries = readArray(value, "tools", warnings);
  if (!entries) return [];
  const tools: ToolDefinitionInput[] = [];
  for (let index = 0; index < entries.length; index += 1) {
    const location = `tools[${index}]`;
    const item = parseRecord(entries[index], location, warnings);
    if (!item) continue;
    const type = readString(item.type);
    if (type !== "function") {
      warning(warnings, location, `non-function Responses tool type ${JSON.stringify(type ?? "(missing)")} is not representable`);
      continue;
    }
    const name = readString(item.name);
    if (!name || !name.trim()) {
      warning(warnings, location, "function definition has no name; it was not added as a stub");
      continue;
    }
    const description = readString(item.description);
    if (item.description !== undefined && description === undefined) warning(warnings, location, "function description was not a string");
    tools.push({
      name,
      ...(description !== undefined ? { description } : {}),
      ...(Object.hasOwn(item, "parameters") ? { parameters: item.parameters } : {}),
    });
  }
  return tools;
}

function parseResponsesFunctionCall(
  item: Record<string, unknown>,
  location: string,
  toolNames: Map<string, string>,
  warnings: string[],
): TracingHistoryMessage | undefined {
  const callId = readString(item.call_id);
  const name = readString(item.name);
  if (!callId || !name) {
    warning(warnings, location, "function_call has no wire call_id or name and cannot be represented");
    return undefined;
  }
  toolNames.set(callId, name);
  if (typeof item.id === "string" && item.id.length > 0) {
    warning(warnings, location, `Responses item id ${JSON.stringify(item.id)} is separate from call_id and is not represented in the canonical history type`);
  }
  const toolCall: TracingHistoryPart = {
    type: "toolCall",
    id: callId,
    name,
    arguments: parseArguments(item.arguments, `${location}.arguments`, warnings),
  };
  return { role: "assistant", content: [toolCall] };
}

function parseResponsesReasoning(
  item: Record<string, unknown>,
  location: string,
  warnings: string[],
): TracingHistoryMessage | undefined {
  const parts: TracingHistoryPart[] = [];
  const summary = item.summary;
  if (typeof summary === "string") parts.push({ type: "thinking", thinking: summary });
  else if (summary !== undefined) {
    const entries = readArray(summary, `${location}.summary`, warnings);
    if (entries) {
      for (let index = 0; index < entries.length; index += 1) {
        const entryLocation = `${location}.summary[${index}]`;
        const entry = parseRecord(entries[index], entryLocation, warnings);
        if (!entry) continue;
        const type = readString(entry.type);
        const text = readString(entry.text);
        if ((type === "summary_text" || type === "summary") && text !== undefined) {
          parts.push({ type: "thinking", thinking: text });
        } else {
          warning(warnings, entryLocation, `unsupported reasoning summary type ${JSON.stringify(type ?? "(missing)")}`);
        }
      }
    }
  }
  const content = item.content;
  if (content !== undefined) {
    const entries = readArray(content, `${location}.content`, warnings);
    if (entries) {
      for (let index = 0; index < entries.length; index += 1) {
        const part = parseContentPart(entries[index], `${location}.content[${index}]`, warnings);
        if (part) parts.push(part);
      }
    }
  }
  if (typeof item.encrypted_content === "string") {
    if (item.encrypted_content.length > 0) parts.push({ type: "redactedThinking", data: item.encrypted_content });
  } else if (item.encrypted_content !== undefined) {
    warning(warnings, `${location}.encrypted_content`, "encrypted reasoning content was not a string");
  }
  if (!parts.length) {
    warning(warnings, location, "reasoning item contained no representable summary or encrypted content");
    return undefined;
  }
  return { role: "assistant", content: parts };
}

function parseResponsesMessage(
  item: Record<string, unknown>,
  location: string,
  systemPrompt: string[],
  messages: TracingHistoryMessage[],
  warnings: string[],
  sawConversationMessage: { value: boolean },
): void {
  const role = readString(item.role);
  if (role !== "user" && role !== "assistant" && role !== "system" && role !== "developer") {
    warning(warnings, location, `unsupported Responses message role ${JSON.stringify(role ?? "(missing)")}`);
    return;
  }
  if (role === "system" || role === "developer") {
    if (sawConversationMessage.value) warning(warnings, location, `${role} prompt occurred after conversation content; canonical systemPrompt moves it to the front`);
    if (item.content === undefined || item.content === null) {
      warning(warnings, location, `${role} message has no content`);
      return;
    }
    addSystemPrompt(systemPrompt, item.content, `${location}.content`, warnings);
    return;
  }
  sawConversationMessage.value = true;
  if (role === "user" && (item.content === undefined || item.content === null)) {
    warning(warnings, location, "user message has no content");
    return;
  }
  const content = normalizeContent(item.content, `${location}.content`, warnings);
  if (role === "assistant" && content.length === 0) return;
  messages.push({ role, content });
}

function normalizeResponses(root: Record<string, unknown>, api: string): NormalizedRequest {
  const warnings: string[] = [];
  const systemPrompt: string[] = [];
  const messages: TracingHistoryMessage[] = [];
  const toolNames = new Map<string, string>();
  const rawInput = responsesInputSchema.safeParse(root.input);
  if (!rawInput.success) throw new Error(`Invalid ${api} request: expected input to be a string or array`);

  if (typeof root.instructions === "string") {
    if (root.instructions.length > 0) systemPrompt.push(root.instructions);
  } else if (root.instructions !== undefined) {
    warning(warnings, "instructions", "instructions was not a string and could not be replayed");
  }

  let previousResponseId: string | undefined;
  if (root.previous_response_id !== undefined) {
    const id = readString(root.previous_response_id);
    if (id && id.length > 0) previousResponseId = id;
    else warning(warnings, "previous_response_id", "delta anchor was not a non-empty string");
  }

  const sawConversationMessage = { value: false };
  const input = rawInput.data;
  if (typeof input === "string") {
    messages.push({ role: "user", content: input });
  } else {
    for (let index = 0; index < input.length; index += 1) {
      const location = `input[${index}]`;
      const item = parseRecord(input[index], location, warnings);
      if (!item) continue;
      const type = readString(item.type);
      if (type === "message" || type === undefined) {
        if (type === undefined && item.role === undefined) {
          warning(warnings, location, "input item has no type or role");
          continue;
        }
        parseResponsesMessage(item, location, systemPrompt, messages, warnings, sawConversationMessage);
        continue;
      }
      if (type === "reasoning") {
        sawConversationMessage.value = true;
        const reasoning = parseResponsesReasoning(item, location, warnings);
        if (reasoning) messages.push(reasoning);
        continue;
      }
      if (type === "function_call" || type === "custom_tool_call") {
        sawConversationMessage.value = true;
        const call = type === "function_call"
          ? parseResponsesFunctionCall(item, location, toolNames, warnings)
          : parseResponsesCustomToolCall(item, location, toolNames, warnings);
        if (call) messages.push(call);
        continue;
      }
      if (type === "function_call_output" || type === "custom_tool_call_output") {
        sawConversationMessage.value = true;
        const callId = readString(item.call_id);
        if (!callId) {
          warning(warnings, location, `${type} has no wire call_id and cannot be represented`);
          continue;
        }
        if (item.output === undefined || item.output === null) {
          warning(warnings, location, `${type} has no output; it was not represented as an empty successful result`);
          continue;
        }
        const content = normalizeContent(item.output, `${location}.output`, warnings);
        const toolName = toolNames.get(callId);
        messages.push({
          role: "toolResult",
          content,
          toolCallId: callId,
          ...(toolName !== undefined ? { toolName } : {}),
          ...(typeof item.status === "string" && item.status === "failed" ? { isError: true } : {}),
        });
        if (typeof item.id === "string" && item.id.length > 0) {
          warning(warnings, location, `output item id ${JSON.stringify(item.id)} is not represented in the canonical history type`);
        }
        continue;
      }
      warning(warnings, location, `unsupported Responses input item type ${JSON.stringify(type)}`);
    }
  }

  const result: NormalizedRequest = {
    systemPrompt,
    messages,
    tools: parseResponsesTools(root.tools, warnings),
    delta: previousResponseId !== undefined,
    ...(previousResponseId !== undefined ? { previousResponseId } : {}),
    warnings,
  };
  return normalizedRequestSchema.parse(result);
}

function parseResponsesCustomToolCall(
  item: Record<string, unknown>,
  location: string,
  toolNames: Map<string, string>,
  warnings: string[],
): TracingHistoryMessage | undefined {
  const callId = readString(item.call_id);
  const name = readString(item.name);
  if (!callId || !name) {
    warning(warnings, location, "custom_tool_call has no wire call_id or name and cannot be represented");
    return undefined;
  }
  toolNames.set(callId, name);
  if (typeof item.id === "string" && item.id.length > 0) {
    warning(warnings, location, `Responses item id ${JSON.stringify(item.id)} is separate from call_id and is not represented in the canonical history type`);
  }
  warning(warnings, location, "custom tool call was represented as a canonical toolCall");
  if (item.input === undefined) warning(warnings, `${location}.input`, "custom tool input was not supplied");
  return {
    role: "assistant",
    content: [{
      type: "toolCall",
      id: callId,
      name,
      arguments: item.input,
    }],
  };
}

/** Normalize a captured OpenAI request body to the plugin's provider-visible history contract. */
export function normalizeOpenAI(payload: unknown, api: string): NormalizedRequest {
  if (api === "openai-completions") return normalizeChatCompletions(parseRoot(payload, api));
  if (api === "openai-responses" || api === "openai-codex-responses") {
    return normalizeResponses(parseRoot(payload, api), api);
  }
  throw new Error(`Unsupported OpenAI API ${JSON.stringify(api)}`);
}
