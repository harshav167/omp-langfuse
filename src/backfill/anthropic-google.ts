import { z } from "@oh-my-pi/pi-coding-agent";
import type { ToolDefinitionInput, TracingHistoryMessage, TracingHistoryPart } from "../index.ts";
import { normalizedRequestSchema as canonicalRequestSchema, readArray, readRecord, type NormalizedRequest } from "./types.ts";

const stringSchema = z.string();
const booleanSchema = z.boolean();


function emptyRequest(warnings: string[]): NormalizedRequest {
  return { systemPrompt: [], messages: [], tools: [], delta: false, warnings };
}

function warn(warnings: string[], message: string): void {
  warnings.push(message);
}


function has(record: Record<string, unknown>, key: string): boolean {
  return Object.hasOwn(record, key);
}

function fieldString(
  record: Record<string, unknown>,
  key: string,
  path: string,
  warnings: string[],
  required = false,
): string | undefined {
  if (!has(record, key)) {
    if (required) warn(warnings, `Missing required string ${path}.${key}.`);
    return undefined;
  }
  const parsed = stringSchema.safeParse(record[key]);
  if (!parsed.success) {
    warn(warnings, `Invalid string ${path}.${key}; the field was not projected.`);
    return undefined;
  }
  return parsed.data;
}

function fieldBoolean(
  record: Record<string, unknown>,
  key: string,
  path: string,
  warnings: string[],
): boolean | undefined {
  if (!has(record, key)) return undefined;
  const parsed = booleanSchema.safeParse(record[key]);
  if (!parsed.success) {
    warn(warnings, `Invalid boolean ${path}.${key}; the field was not projected.`);
    return undefined;
  }
  return parsed.data;
}

function fieldRecord(
  record: Record<string, unknown>,
  key: string,
  path: string,
  warnings: string[],
  required = false,
): Record<string, unknown> | undefined {
  if (!has(record, key)) {
    if (required) warn(warnings, `Missing required object ${path}.${key}.`);
    return undefined;
  }
  const parsed = readRecord(record[key]);
  if (!parsed) warn(warnings, `Invalid object ${path}.${key}; the field was not projected.`);
  return parsed;
}

function fieldArray(
  record: Record<string, unknown>,
  key: string,
  path: string,
  warnings: string[],
  required = false,
): unknown[] | undefined {
  if (!has(record, key)) {
    if (required) warn(warnings, `Missing required array ${path}.${key}.`);
    return undefined;
  }
  const parsed = readArray(record[key]);
  if (!parsed) warn(warnings, `Invalid array ${path}.${key}; its entries were not projected.`);
  return parsed;
}

function warnUnknownFields(
  record: Record<string, unknown>,
  known: readonly string[],
  path: string,
  warnings: string[],
): void {
  const unknown = Object.keys(record).filter(key => !known.includes(key));
  if (unknown.length > 0) {
    warn(warnings, `Unprojected provider fields at ${path}: ${unknown.join(", ")}.`);
  }
}

function warnPresentFields(
  record: Record<string, unknown>,
  fields: readonly string[],
  path: string,
  warnings: string[],
): void {
  const present = fields.filter(field => has(record, field));
  if (present.length > 0) {
    warn(warnings, `Provider fields outside the normalized history at ${path}: ${present.join(", ")}.`);
  }
}

function warnSignature(
  value: unknown,
  path: string,
  warnings: string[],
): void {
  const signature = stringSchema.safeParse(value);
  if (!signature.success) {
    warn(warnings, `Invalid provider signature metadata at ${path}; it was not projected.`);
    return;
  }
  if (signature.data === "skip_thought_signature_validator") {
    warn(warnings, `Provider thought-signature bypass sentinel at ${path} was not projected as reasoning.`);
    return;
  }
  warn(warnings, `Opaque provider signature/replay metadata at ${path} is not canonical reasoning and was not projected.`);
}

function parseAnthropicSystem(
  value: unknown,
  path: string,
  systemPrompt: string[],
  warnings: string[],
): void {
  const directText = stringSchema.safeParse(value);
  if (directText.success) {
    systemPrompt.push(directText.data);
    return;
  }
  const blocks = readArray(value);
  if (!blocks) {
    warn(warnings, `Invalid Anthropic system content at ${path}; expected text or an array of text blocks.`);
    return;
  }
  for (const [index, rawBlock] of blocks.entries()) {
    const blockPath = `${path}[${index}]`;
    const block = readRecord(rawBlock);
    if (!block) {
      warn(warnings, `Invalid Anthropic system block at ${blockPath}; it was not projected.`);
      continue;
    }
    const type = fieldString(block, "type", blockPath, warnings, true);
    const text = fieldString(block, "text", blockPath, warnings, type === "text");
    if (type !== "text" || text === undefined) {
      if (type !== "text") warn(warnings, `Unsupported Anthropic system block ${type ?? "(missing type)"} at ${blockPath}.`);
      continue;
    }
    warnUnknownFields(block, ["type", "text", "cache_control"], blockPath, warnings);
    systemPrompt.push(text);
  }
}

function anthropicImage(
  block: Record<string, unknown>,
  path: string,
  warnings: string[],
): TracingHistoryPart | undefined {
  const source = fieldRecord(block, "source", path, warnings, true);
  if (!source) return undefined;
  const sourceType = fieldString(source, "type", `${path}.source`, warnings, true);
  if (sourceType !== "base64") {
    warn(warnings, `Anthropic image source ${sourceType ?? "(missing type)"} at ${path}.source is not inline base64 and cannot be represented.`);
    return undefined;
  }
  const data = fieldString(source, "data", `${path}.source`, warnings, true);
  const mimeType = fieldString(source, "media_type", `${path}.source`, warnings, true);
  warnUnknownFields(source, ["type", "data", "media_type"], `${path}.source`, warnings);
  return data !== undefined && mimeType !== undefined ? { type: "image", data, mimeType } : undefined;
}

function anthropicTool(
  rawTool: unknown,
  path: string,
  warnings: string[],
): ToolDefinitionInput | undefined {
  const tool = readRecord(rawTool);
  if (!tool) {
    warn(warnings, `Invalid Anthropic tool declaration at ${path}; it was not projected.`);
    return undefined;
  }
  const name = fieldString(tool, "name", path, warnings, true);
  if (name === undefined || name.length === 0) {
    if (name === "") warn(warnings, `Empty Anthropic tool name at ${path}; the declaration was not projected.`);
    return undefined;
  }
  const description = fieldString(tool, "description", path, warnings);
  const schema = fieldRecord(tool, "input_schema", path, warnings, true);
  warnUnknownFields(tool, ["name", "description", "input_schema", "cache_control"], path, warnings);
  return {
    name,
    ...(description !== undefined && { description }),
    ...(schema !== undefined && { parameters: schema }),
  };
}

function anthropicBlock(
  rawBlock: unknown,
  role: "user" | "assistant" | "system",
  path: string,
  warnings: string[],
  toolNames: Map<string, string>,
): { parts: TracingHistoryPart[]; result?: TracingHistoryMessage } {
  const block = readRecord(rawBlock);
  if (!block) {
    warn(warnings, `Invalid Anthropic content block at ${path}; it was not projected.`);
    return { parts: [] };
  }
  const type = fieldString(block, "type", path, warnings, true);
  if (!type) return { parts: [] };

  if (type === "text") {
    const text = fieldString(block, "text", path, warnings, true);
    warnUnknownFields(block, ["type", "text", "cache_control"], path, warnings);
    return { parts: text === undefined ? [] : [{ type: "text", text }] };
  }
  if (type === "image") {
    const image = anthropicImage(block, path, warnings);
    warnUnknownFields(block, ["type", "source", "cache_control"], path, warnings);
    return { parts: image ? [image] : [] };
  }
  if (type === "thinking") {
    if (role !== "assistant") warn(warnings, `Anthropic thinking block at ${path} appeared in ${role} content.`);
    const thinking = fieldString(block, "thinking", path, warnings, true);
    const parts: TracingHistoryPart[] = thinking === undefined ? [] : [{ type: "thinking", thinking }];
    if (has(block, "signature")) warnSignature(block.signature, `${path}.signature`, warnings);
    else warn(warnings, `Missing required Anthropic thinking signature at ${path}.`);
    warnUnknownFields(block, ["type", "thinking", "signature", "cache_control"], path, warnings);
    return { parts };
  }
  if (type === "redacted_thinking") {
    const data = fieldString(block, "data", path, warnings, true);
    warnUnknownFields(block, ["type", "data", "cache_control"], path, warnings);
    return { parts: data === undefined ? [] : [{ type: "redactedThinking", data }] };
  }
  if (type === "tool_use" || type === "server_tool_use") {
    if (role !== "assistant") {
      warn(warnings, `Anthropic ${type} block at ${path} appeared in ${role} content.`);
      return { parts: [] };
    }
    const id = fieldString(block, "id", path, warnings, true);
    const name = fieldString(block, "name", path, warnings, true);
    const argumentsValue = has(block, "input") ? block.input : undefined;
    if (!has(block, "input")) warn(warnings, `Missing ${path}.input; tool arguments remain unavailable.`);
    warnUnknownFields(block, ["type", "id", "name", "input", "cache_control"], path, warnings);
    if (type === "server_tool_use") {
      warn(warnings, `Anthropic server-side tool call at ${path} is represented as a toolCall; execution is not represented.`);
    }
    if (id === undefined || name === undefined) return { parts: [] };
    toolNames.set(id, name);
    return { parts: [{ type: "toolCall", id, name, arguments: argumentsValue }] };
  }
  if (type === "tool_result" || type === "web_search_tool_result" || type === "tool_search_tool_result") {
    if (role !== "user") warn(warnings, `Anthropic ${type} block at ${path} appeared in ${role} content.`);
    const id = fieldString(block, "tool_use_id", path, warnings, true);
    const content = has(block, "content") ? block.content : undefined;
    const isError = fieldBoolean(block, "is_error", path, warnings);
    warnUnknownFields(block, ["type", "tool_use_id", "content", "is_error", "cache_control"], path, warnings);
    if (type !== "tool_result") {
      warn(warnings, `Anthropic provider-side tool result ${type} at ${path} is represented as a toolResult.`);
    }
    if (id === undefined) return { parts: [] };
    const messageContent = anthropicToolResultContent(content, `${path}.content`, warnings);
    return {
      parts: [],
      result: {
        role: "toolResult",
        content: messageContent,
        toolCallId: id,
        ...(toolNames.has(id) && { toolName: toolNames.get(id) }),
        ...(isError !== undefined && { isError }),
      },
    };
  }
  if (type === "compaction") {
    const content = fieldString(block, "content", path, warnings, true);
    const parts: TracingHistoryPart[] = content === undefined ? [] : [{ type: "text", text: content }];
    if (content !== undefined) warn(warnings, `Anthropic compaction summary at ${path} was flattened to text; provider replay metadata is not represented.`);
    if (has(block, "encrypted_content") && block.encrypted_content !== null) {
      const encryptedContent = fieldString(block, "encrypted_content", path, warnings);
      if (encryptedContent !== undefined) {
        warn(warnings, `Anthropic compaction encrypted_content at ${path} is opaque provider replay state, not canonical reasoning.`);
      }
    }
    if (has(block, "signature")) {
      warnSignature(block.signature, `${path}.signature`, warnings);
    }
    warnUnknownFields(block, ["type", "content", "encrypted_content", "signature", "cache_control"], path, warnings);
    return { parts };
  }

  if (type === "fallback" || type === "tool_addition" || type === "tool_removal") {
    warn(warnings, `Unsupported Anthropic provider control block ${type} at ${path}; its fields are not represented.`);
    return { parts: [] };
  }
  warn(warnings, `Unsupported Anthropic content block ${type} at ${path}; it was not projected.`);
  return { parts: [] };
}

function anthropicToolResultContent(
  value: unknown,
  path: string,
  warnings: string[],
): string | TracingHistoryPart[] {
  const text = stringSchema.safeParse(value);
  if (text.success) return text.data;
  const blocks = readArray(value);
  if (!blocks) {
    if (value !== undefined) warn(warnings, `Invalid Anthropic tool result content at ${path}; it was not projected.`);
    return [];
  }
  const parts: TracingHistoryPart[] = [];
  for (const [index, rawBlock] of blocks.entries()) {
    const blockPath = `${path}[${index}]`;
    const block = readRecord(rawBlock);
    if (!block) {
      warn(warnings, `Invalid Anthropic tool result block at ${blockPath}; it was not projected.`);
      continue;
    }
    const type = fieldString(block, "type", blockPath, warnings, true);
    if (type === "text") {
      const valueText = fieldString(block, "text", blockPath, warnings, true);
      if (valueText !== undefined) parts.push({ type: "text", text: valueText });
      warnUnknownFields(block, ["type", "text", "cache_control"], blockPath, warnings);
    } else if (type === "image") {
      const image = anthropicImage(block, blockPath, warnings);
      if (image) parts.push(image);
      warnUnknownFields(block, ["type", "source", "cache_control"], blockPath, warnings);
    } else {
      warn(warnings, `Unsupported Anthropic tool result block ${type ?? "(missing type)"} at ${blockPath}.`);
    }
  }
  return parts;
}

function normalizeAnthropic(
  payload: Record<string, unknown>,
  warnings: string[],
): Omit<NormalizedRequest, "warnings"> {
  const systemPrompt: string[] = [];
  const messages: TracingHistoryMessage[] = [];
  const tools: ToolDefinitionInput[] = [];
  const toolNames = new Map<string, string>();

  if (has(payload, "system")) parseAnthropicSystem(payload.system, "$.system", systemPrompt, warnings);
  const rawMessages = fieldArray(payload, "messages", "$", warnings, true) ?? [];
  for (const [index, rawMessage] of rawMessages.entries()) {
    const path = `$.messages[${index}]`;
    const message = readRecord(rawMessage);
    if (!message) {
      warn(warnings, `Invalid Anthropic message at ${path}; it was not projected.`);
      continue;
    }
    const role = fieldString(message, "role", path, warnings, true);
    if (role !== "user" && role !== "assistant" && role !== "system") {
      warn(warnings, `Unsupported Anthropic message role ${role ?? "(missing role)"} at ${path}; its content was not projected.`);
      continue;
    }
    warnUnknownFields(message, ["role", "content"], path, warnings);
    if (!has(message, "content")) {
      warn(warnings, `Missing ${path}.content; the message was not projected.`);
      continue;
    }
    const directText = stringSchema.safeParse(message.content);
    if (directText.success) {
      if (role === "system") systemPrompt.push(directText.data);
      else messages.push({ role, content: directText.data });
      continue;
    }
    const blocks = readArray(message.content);
    if (!blocks) {
      warn(warnings, `Invalid Anthropic message content at ${path}.content; it was not projected.`);
      continue;
    }
    const ordinaryParts: TracingHistoryPart[] = [];
    const messagesBefore = messages.length;
    const flushOrdinary = () => {
      if (ordinaryParts.length === 0) return;
      if (role === "system") {
        for (const part of ordinaryParts) {
          if (part.type === "text") systemPrompt.push(part.text);
          else warn(warnings, `Anthropic system content ${part.type} at ${path}.content cannot be represented as a system prompt.`);
        }
      } else {
        messages.push({ role, content: ordinaryParts.splice(0) });
      }
    };
    for (const [blockIndex, rawBlock] of blocks.entries()) {
      const projected = anthropicBlock(rawBlock, role, `${path}.content[${blockIndex}]`, warnings, toolNames);
      if (projected.result) {
        flushOrdinary();
        messages.push(projected.result);
      } else {
        ordinaryParts.push(...projected.parts);
      }
    }
    flushOrdinary();
    if (role !== "system" && messages.length === messagesBefore) messages.push({ role, content: [] });
  }

  const rawTools = fieldArray(payload, "tools", "$", warnings) ?? [];
  for (const [index, rawTool] of rawTools.entries()) {
    const tool = anthropicTool(rawTool, `$.tools[${index}]`, warnings);
    if (tool) tools.push(tool);
  }
  warnUnknownFields(payload, ["model", "messages", "system", "tools", "max_tokens", "stream", "thinking", "output_config", "metadata", "tool_choice", "stop_sequences", "temperature", "top_p", "top_k", "context_management", "speed", "fallbacks", "anthropic_beta", "compaction", "service_tier", "cache_control"], "$", warnings);
  return { systemPrompt, messages, tools, delta: false };
}

function googleImage(
  part: Record<string, unknown>,
  path: string,
  warnings: string[],
): TracingHistoryPart | undefined {
  if (has(part, "inlineData") && has(part, "fileData")) {
    warn(warnings, `Google content part at ${path} contains both inlineData and fileData; only inlineData was projected.`);
  }
  if (has(part, "inlineData")) {
    const inline = fieldRecord(part, "inlineData", path, warnings, true);
    if (!inline) return undefined;
    const data = fieldString(inline, "data", `${path}.inlineData`, warnings, true);
    const mimeType = fieldString(inline, "mimeType", `${path}.inlineData`, warnings, true);
    warnUnknownFields(inline, ["data", "mimeType"], `${path}.inlineData`, warnings);
    if (!mimeType?.startsWith("image/")) {
      warn(warnings, `Unsupported non-image Google inlineData at ${path}; it cannot be represented as an image.`);
      return undefined;
    }
    return data !== undefined ? { type: "image", data, mimeType } : undefined;
  }
  if (has(part, "fileData")) {
    const file = fieldRecord(part, "fileData", path, warnings, true);
    if (!file) return undefined;
    const uri = fieldString(file, "fileUri", `${path}.fileData`, warnings, true);
    const mimeType = fieldString(file, "mimeType", `${path}.fileData`, warnings);
    warnUnknownFields(file, ["fileUri", "mimeType"], `${path}.fileData`, warnings);
    if (uri) {
      const match = /^data:(image\/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/]*={0,2})$/.exec(uri);
      if (match) return { type: "image", mimeType: match[1]!, data: match[2]! };
    }
    warn(warnings, `Google remote fileData (${mimeType ?? "unknown MIME"}) at ${path} cannot be represented without fetching the referenced file.`);
    return undefined;
  }
  return undefined;
}

function googleToolResult(
  rawFunctionResponse: unknown,
  path: string,
  warnings: string[],
): TracingHistoryMessage | undefined {
  const fn = readRecord(rawFunctionResponse);
  if (!fn) {
    warn(warnings, `Invalid Google functionResponse at ${path}; it was not projected.`);
    return undefined;
  }
  const name = fieldString(fn, "name", path, warnings);
  const id = fieldString(fn, "id", path, warnings);
  const response = fieldRecord(fn, "response", path, warnings, true);
  const rawParts = fieldArray(fn, "parts", path, warnings);
  warnUnknownFields(fn, ["name", "id", "response", "parts"], path, warnings);

  let text: string | undefined;
  let isError: boolean | undefined;
  if (response) {
    if (Object.keys(response).length === 1 && typeof response.output === "string") {
      text = response.output;
    } else if (Object.keys(response).length === 1 && typeof response.error === "string") {
      text = response.error;
      isError = true;
    } else {
      try {
        text = JSON.stringify(response);
      } catch {
        warn(warnings, `Google functionResponse value at ${path}.response is not serializable; it was not projected.`);
      }
      if (has(response, "error")) isError = true;
    }
  }

  const resultParts: TracingHistoryPart[] = [];
  if (text !== undefined) resultParts.push({ type: "text", text });
  for (const [index, rawPart] of (rawParts ?? []).entries()) {
    const partPath = `${path}.parts[${index}]`;
    const part = readRecord(rawPart);
    if (!part) {
      warn(warnings, `Invalid Google functionResponse part at ${partPath}; it was not projected.`);
      continue;
    }
    if (has(part, "text")) {
      const partText = fieldString(part, "text", partPath, warnings);
      if (partText !== undefined) resultParts.push({ type: "text", text: partText });
    }
    const image = googleImage(part, partPath, warnings);
    if (image) resultParts.push(image);
    const known = ["text", "inlineData", "fileData"];
    if (Object.keys(part).every(key => !known.includes(key))) {
      warn(warnings, `Unsupported Google functionResponse part at ${partPath}; it was not projected.`);
    } else {
      warnUnknownFields(part, known, partPath, warnings);
    }
  }

  if (name === undefined && id === undefined && response === undefined && resultParts.length === 0) return undefined;
  return {
    role: "toolResult",
    content: rawParts === undefined && resultParts.length === 1 && resultParts[0]!.type === "text"
      ? resultParts[0]!.text
      : resultParts,
    ...(id !== undefined && { toolCallId: id }),
    ...(name !== undefined && { toolName: name }),
    ...(isError !== undefined && { isError }),
  };
}

function googlePart(
  rawPart: unknown,
  role: "user" | "assistant",
  path: string,
  warnings: string[],
  calls: Map<string, string>,
  inferredIds: Set<string>,
): TracingHistoryPart[] {
  const part = readRecord(rawPart);
  if (!part) {
    warn(warnings, `Invalid Google content part at ${path}; it was not projected.`);
    return [];
  }
  const projected: TracingHistoryPart[] = [];
  if (has(part, "text")) {
    const text = fieldString(part, "text", path, warnings);
    if (text !== undefined) {
      const thought = fieldBoolean(part, "thought", path, warnings);
      projected.push(thought === true ? { type: "thinking", thinking: text } : { type: "text", text });
    }
  } else if (has(part, "thought") && part.thought !== false) {
    warn(warnings, `Google thought marker without text at ${path} has no canonical content.`);
  }

  if (has(part, "inlineData") || has(part, "fileData")) {
    const image = googleImage(part, path, warnings);
    if (image) projected.push(image);
  }

  if (has(part, "functionCall")) {
    const fn = fieldRecord(part, "functionCall", path, warnings, true);
    if (fn) {
      const name = fieldString(fn, "name", `${path}.functionCall`, warnings, true);
      let id = fieldString(fn, "id", `${path}.functionCall`, warnings);
      let args: unknown;
      if (has(fn, "args")) {
        const parsedArgs = readRecord(fn.args);
        if (parsedArgs) args = parsedArgs;
        else warn(warnings, `Invalid Google functionCall args at ${path}.functionCall.args; arguments were not projected.`);
      } else {
        warn(warnings, `Missing ${path}.functionCall.args; tool arguments remain unavailable.`);
      }
      warnUnknownFields(fn, ["name", "id", "args"], `${path}.functionCall`, warnings);
      if (role !== "assistant") {
        warn(warnings, `Google functionCall at ${path} appeared in user content and was not projected as an executable call.`);
      } else if (name !== undefined && name !== "") {
        if (id === undefined || id === "") {
          id = `google-history:${path}:${encodeURIComponent(name)}`;
          inferredIds.add(id);
        }
        calls.set(id, name);
        projected.push({ type: "toolCall", id, name, arguments: args });
      } else {
        warn(warnings, `Google functionCall at ${path} lacks a representable name; its arguments were not projected.`);
      }
    }
  }

  if (has(part, "functionResponse")) {
    warn(warnings, `Google functionResponse at ${path} must be projected as a separate toolResult and was not accepted in ordinary ${role} content.`);
  }
  if (has(part, "thoughtSignature")) {
    warnSignature(part.thoughtSignature, `${path}.thoughtSignature`, warnings);
  }
  warnUnknownFields(part, ["text", "thought", "thoughtSignature", "inlineData", "fileData", "functionCall", "functionResponse"], path, warnings);
  return projected;
}

function normalizeGoogleSystemInstruction(
  value: unknown,
  path: string,
  systemPrompt: string[],
  warnings: string[],
): void {
  const instruction = readRecord(value);
  if (!instruction) {
    warn(warnings, `Invalid Google systemInstruction at ${path}; it was not projected.`);
    return;
  }
  const parts = fieldArray(instruction, "parts", path, warnings, true) ?? [];
  warnUnknownFields(instruction, ["role", "parts"], path, warnings);
  for (const [index, rawPart] of parts.entries()) {
    const partPath = `${path}.parts[${index}]`;
    const part = readRecord(rawPart);
    if (!part) {
      warn(warnings, `Invalid Google system instruction part at ${partPath}; it was not projected.`);
      continue;
    }
    const text = fieldString(part, "text", partPath, warnings, true);
    if (text !== undefined) systemPrompt.push(text);
    warnUnknownFields(part, ["text"], partPath, warnings);
  }
}


function normalizeGoogleTools(
  value: unknown,
  path: string,
  warnings: string[],
): ToolDefinitionInput[] {
  const groups = readArray(value);
  if (!groups) {
    warn(warnings, `Invalid Google tools at ${path}; tools were not projected.`);
    return [];
  }
  const tools: ToolDefinitionInput[] = [];
  for (const [index, rawGroup] of groups.entries()) {
    const group = readRecord(rawGroup);
    if (!group) {
      warn(warnings, `Invalid Google tool group at ${path}[${index}]; it was not projected.`);
      continue;
    }
    const declarations = fieldArray(group, "functionDeclarations", `${path}[${index}]`, warnings, true) ?? [];
    warnUnknownFields(group, ["functionDeclarations"], `${path}[${index}]`, warnings);
    for (const [declarationIndex, rawDeclaration] of declarations.entries()) {
      const declarationPath = `${path}[${index}].functionDeclarations[${declarationIndex}]`;
      const declaration = readRecord(rawDeclaration);
      if (!declaration) {
        warn(warnings, `Invalid Google function declaration at ${declarationPath}; it was not projected.`);
        continue;
      }
      const name = fieldString(declaration, "name", declarationPath, warnings, true);
      if (name === undefined || name === "") {
        if (name === "") warn(warnings, `Empty Google tool name at ${declarationPath}; the declaration was not projected.`);
        continue;
      }
      const description = fieldString(declaration, "description", declarationPath, warnings);
      const parametersJsonSchema = fieldRecord(declaration, "parametersJsonSchema", declarationPath, warnings);
      const parameters = fieldRecord(declaration, "parameters", declarationPath, warnings);
      if (parametersJsonSchema && parameters) {
        warn(warnings, `Both Google tool schema fields are present at ${declarationPath}; parametersJsonSchema was selected.`);
      }
      const schema = parametersJsonSchema ?? parameters;
      if (schema === undefined) warn(warnings, `Missing Google input schema at ${declarationPath}; the declaration has no parameters projection.`);
      warnUnknownFields(declaration, ["name", "description", "parametersJsonSchema", "parameters"], declarationPath, warnings);
      tools.push({ name, ...(description !== undefined && { description }), ...(schema !== undefined && { parameters: schema }) });
    }
  }
  return tools;
}

function normalizeGoogle(
  payload: Record<string, unknown>,
  api: string,
  warnings: string[],
): Omit<NormalizedRequest, "warnings"> & { previousResponseId?: string } {
  const isCloudCode = api === "google-gemini-cli";
  const request = isCloudCode ? fieldRecord(payload, "request", "$", warnings, true) : undefined;
  if (isCloudCode && !request) return { systemPrompt: [], messages: [], tools: [], delta: false };
  const body = request ?? payload;
  const prefix = request ? "$.request" : "$";
  const config = !request ? fieldRecord(payload, "config", "$", warnings) : undefined;
  const contents = fieldArray(body, "contents", prefix, warnings, true) ?? [];
  const systemPrompt: string[] = [];
  const messages: TracingHistoryMessage[] = [];
  const tools: ToolDefinitionInput[] = [];
  const calls = new Map<string, string>();
  const inferredIds = new Set<string>();

  const systemInstruction = has(body, "systemInstruction")
    ? body.systemInstruction
    : config && has(config, "systemInstruction")
      ? config.systemInstruction
      : has(payload, "systemInstruction")
        ? payload.systemInstruction
        : undefined;
  if (systemInstruction !== undefined) {
    const systemPath = has(body, "systemInstruction") ? `${prefix}.systemInstruction` : config && has(config, "systemInstruction") ? "$.config.systemInstruction" : "$.systemInstruction";
    normalizeGoogleSystemInstruction(systemInstruction, systemPath, systemPrompt, warnings);
  }

  const toolValue = has(body, "tools") ? body.tools : config && has(config, "tools") ? config.tools : has(payload, "tools") ? payload.tools : undefined;
  if (toolValue !== undefined) {
    const toolPath = has(body, "tools") ? `${prefix}.tools` : config && has(config, "tools") ? "$.config.tools" : "$.tools";
    tools.push(...normalizeGoogleTools(toolValue, toolPath, warnings));
  }

  for (const [index, rawContent] of contents.entries()) {
    const path = `${prefix}.contents[${index}]`;
    const content = readRecord(rawContent);
    if (!content) {
      warn(warnings, `Invalid Google content at ${path}; it was not projected.`);
      continue;
    }
    const role = fieldString(content, "role", path, warnings);
    const parts = fieldArray(content, "parts", path, warnings, true) ?? [];
    warnUnknownFields(content, ["role", "parts"], path, warnings);
    let targetRole: "user" | "assistant" | "system";
    if (role === "user") targetRole = "user";
    else if (role === "model" || role === "assistant") targetRole = "assistant";
    else if (role === "system" || role === undefined) {
      targetRole = "system";
      if (role === undefined) warn(warnings, `Google content at ${path} has no role and was treated as system instruction content.`);
    } else {
      warn(warnings, `Unsupported Google content role ${role} at ${path}; its content was not projected.`);
      continue;
    }
    // Responses belong to the preceding assistant call group, not an older
    // unresolved call with the same name.
    if (targetRole === "assistant") calls.clear();

    const pending: TracingHistoryPart[] = [];
    const messagesBefore = messages.length;
    const flush = () => {
      if (!pending.length) return;
      if (targetRole === "system") {
        for (const part of pending) {
          if (part.type === "text") systemPrompt.push(part.text);
          else warn(warnings, `Google system content ${part.type} at ${path} cannot be represented as a system instruction.`);
        }
        pending.length = 0;
      } else {
        messages.push({ role: targetRole, content: pending.splice(0) });
      }
    };
    for (const [partIndex, rawPart] of parts.entries()) {
      const partPath = `${path}.parts[${partIndex}]`;
      const partRecord = readRecord(rawPart);
      if (!partRecord) {
        warn(warnings, `Invalid Google content part at ${partPath}; it was not projected.`);
        continue;
      }
      if (has(partRecord, "functionResponse")) {
        if (targetRole !== "user") {
          warn(warnings, `Google functionResponse at ${partPath} appeared outside user content and was not projected.`);
          continue;
        }
        flush();
        const result = googleToolResult(partRecord.functionResponse, `${partPath}.functionResponse`, warnings);
        if (result) {
          if (!result.toolCallId && result.toolName) {
            const matches = [...calls].filter(([, name]) => name === result.toolName);
            if (matches.length === 1) {
              result.toolCallId = matches[0]![0];
              inferredIds.add(result.toolCallId);
            } else {
              warn(warnings, `Google functionResponse at ${partPath} has no unique preceding call; its identity remains unavailable.`);
            }
          }
          if (result.toolCallId) calls.delete(result.toolCallId);
          messages.push(result);
        }
        const extras = Object.keys(partRecord).filter(key => key !== "functionResponse");
        if (extras.length) warn(warnings, `Unprojected fields alongside functionResponse at ${partPath}: ${extras.join(", ")}.`);
        continue;
      }
      pending.push(...googlePart(rawPart, targetRole === "system" ? "user" : targetRole, partPath, warnings, calls, inferredIds));
    }
    flush();
    if (targetRole !== "system" && messages.length === messagesBefore) messages.push({ role: targetRole, content: [] });
  }

  let previousResponseId: string | undefined;
  if (request) {
    const labels = fieldRecord(request, "labels", "$.request", warnings);
    if (labels) {
      previousResponseId = fieldString(labels, "last_execution_id", "$.request.labels", warnings);
      warnUnknownFields(labels, ["last_execution_id", "last_step_index", "model_enum", "trajectory_id", "used_claude", "used_claude_conservative", "used_claude_model", "used_claude_version"], "$.request.labels", warnings);
      warnPresentFields(labels, ["last_step_index", "model_enum", "trajectory_id", "used_claude", "used_claude_conservative", "used_claude_model", "used_claude_version"], "$.request.labels", warnings);
    }
    warnUnknownFields(request, ["contents", "systemInstruction", "generationConfig", "tools", "toolConfig", "labels", "sessionId"], "$.request", warnings);
    warnPresentFields(request, ["generationConfig", "toolConfig", "sessionId"], "$.request", warnings);
    warnUnknownFields(payload, ["project", "model", "request", "requestId", "requestType", "userAgent"], "$", warnings);
    warnPresentFields(payload, ["requestId", "requestType", "userAgent"], "$", warnings);
  } else {
    warnUnknownFields(payload, ["model", "contents", "config", "systemInstruction", "tools", "toolConfig", "safetySettings", "cachedContent", "serviceTier", "generationConfig"], "$", warnings);
    warnPresentFields(payload, ["generationConfig", "toolConfig", "safetySettings", "cachedContent", "serviceTier"], "$", warnings);
    if (config) warnUnknownFields(config, ["systemInstruction", "tools", "toolConfig", "safetySettings", "cachedContent", "thinkingConfig", "serviceTier", "temperature", "maxOutputTokens", "topP", "topK", "candidateCount", "stopSequences", "presencePenalty", "frequencyPenalty", "seed", "responseMimeType", "responseSchema", "responseJsonSchema", "responseModalities", "abortSignal"], "$.config", warnings);
    if (config) warnPresentFields(config, ["thinkingConfig", "toolConfig", "safetySettings", "cachedContent", "serviceTier", "temperature", "maxOutputTokens", "topP", "topK", "candidateCount", "stopSequences", "presencePenalty", "frequencyPenalty", "seed", "responseMimeType", "responseSchema", "responseJsonSchema", "responseModalities", "abortSignal"], "$.config", warnings);
  }
  return {
    systemPrompt, messages, tools, delta: false,
    ...(previousResponseId !== undefined && { previousResponseId }),
    ...(inferredIds.size > 0 && { inferredToolCallIds: [...inferredIds] }),
  };
}

function finalize(
  partial: Omit<NormalizedRequest, "warnings"> & { previousResponseId?: string },
  warnings: string[],
): NormalizedRequest {
  const candidate = { ...partial, warnings };
  const parsed = canonicalRequestSchema.safeParse(candidate);
  if (!parsed.success) {
    warn(warnings, `Canonical request validation failed: ${parsed.error.message}`);
    return emptyRequest(warnings);
  }
  return parsed.data;
}

/** Project archived Anthropic Messages or Google GenerateContent requests into the shared trace history shape. */
export function normalizeAnthropicGoogle(payload: unknown, api: string): NormalizedRequest {
  const warnings: string[] = [];
  const root = readRecord(payload);
  if (!root) {
    warn(warnings, "Provider request payload must be a JSON object; no history was projected.");
    return emptyRequest(warnings);
  }
  if (api === "anthropic-messages") return finalize(normalizeAnthropic(root, warnings), warnings);
  if (api === "google-generative-ai" || api === "google-vertex" || api === "google-gemini-cli") {
    return finalize(normalizeGoogle(root, api, warnings), warnings);
  }
  warn(warnings, `Unsupported provider API ${JSON.stringify(api)}; no history was projected.`);
  return emptyRequest(warnings);
}
