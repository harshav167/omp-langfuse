import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { normalizeAnthropicGoogle } from "../src/backfill/anthropic-google.ts";

const imageBlock = {
  type: "image",
  source: { type: "base64", media_type: "image/png", data: "aW1hZ2U=" },
};

const googleImagePart = { inlineData: { mimeType: "image/jpeg", data: "amVwZw==" } };

describe("normalizeAnthropicGoogle", () => {
  it("projects Anthropic system, multimodal messages, thinking, and paired tool results", () => {
    const normalized = normalizeAnthropicGoogle(
      {
        model: "claude-example",
        system: [{ type: "text", text: "Use the available tools." }],
        messages: [
          {
            role: "user",
            content: [{ type: "text", text: "Find the image" }, imageBlock],
          },
          {
            role: "assistant",
            content: [
              { type: "thinking", thinking: "I should inspect it.", signature: "signed-thought" },
              { type: "redacted_thinking", data: "encrypted-thought" },
              { type: "compaction", content: "Summary of the prior context.", signature: "compaction-signature", encrypted_content: "opaque-compaction-state" },
              { type: "tool_use", id: "toolu_01", name: "inspect", input: { query: "image" } },
            ],
          },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "toolu_01",
                content: [{ type: "text", text: "Image found" }, imageBlock],
                is_error: false,
              },
            ],
          },
        ],
        tools: [
          {
            name: "inspect",
            description: "Inspect an image",
            input_schema: { type: "object", properties: { query: { type: "string" } } },
          },
        ],
        max_tokens: 1024,
      },
      "anthropic-messages",
    );

    assert.deepEqual(normalized.systemPrompt, ["Use the available tools."]);
    assert.equal(normalized.delta, false);
    assert.deepEqual(normalized.tools, [
      {
        name: "inspect",
        description: "Inspect an image",
        parameters: { type: "object", properties: { query: { type: "string" } } },
      },
    ]);
    assert.deepEqual(normalized.messages, [
      {
        role: "user",
        content: [
          { type: "text", text: "Find the image" },
          { type: "image", mimeType: "image/png", data: "aW1hZ2U=" },
        ],
      },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "I should inspect it." },
          { type: "redactedThinking", data: "encrypted-thought" },
          { type: "text", text: "Summary of the prior context." },
          { type: "toolCall", id: "toolu_01", name: "inspect", arguments: { query: "image" } },
        ],
      },
      {
        role: "toolResult",
        content: [
          { type: "text", text: "Image found" },
          { type: "image", mimeType: "image/png", data: "aW1hZ2U=" },
        ],
        toolCallId: "toolu_01",
        toolName: "inspect",
        isError: false,
      },
    ]);
    assert.ok(normalized.warnings.some(warning => warning.includes("signature/replay metadata")));
    assert.ok(normalized.warnings.some(warning => warning.includes("compaction encrypted_content")));
  });

  it("accepts Anthropic cache markers without making complete semantic history incomplete", () => {
    const cache_control = { type: "ephemeral", ttl: "1h" };
    const normalized = normalizeAnthropicGoogle({
      cache_control,
      system: [{ type: "text", text: "system", cache_control }],
      tools: [{ name: "inspect", input_schema: { type: "object" }, cache_control }],
      messages: [
        { role: "user", content: [{ type: "text", text: "input", cache_control }, { ...imageBlock, cache_control }] },
        { role: "assistant", content: [{ type: "tool_use", id: "actual-call", name: "inspect", input: {}, cache_control }] },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "actual-call", content: [{ type: "text", text: "result", cache_control }], cache_control }] },
      ],
    }, "anthropic-messages");
    assert.deepEqual(normalized.warnings, []);
    assert.equal(normalized.messages[2]?.toolName, "inspect");
    assert.deepEqual(normalized.systemPrompt, ["system"]);
  });

  it("preserves id-less Gemini calls and uniquely paired results with deterministic inferred provenance", () => {
    const payload = { contents: [
      { role: "model", parts: [
        { functionCall: { name: "read", args: { path: "a" } } },
        { functionCall: { name: "grep", args: { pattern: "b" } } },
      ] },
      { role: "user", parts: [
        { functionResponse: { name: "grep", response: { output: "grep-result" } } },
        { functionResponse: { name: "read", response: { output: "read-result" } } },
      ] },
    ] };
    const normalized = normalizeAnthropicGoogle(payload, "google-generative-ai");
    assert.deepEqual(normalized, normalizeAnthropicGoogle(payload, "google-generative-ai"));
    const assistant = normalized.messages[0]!;
    assert.ok(Array.isArray(assistant.content));
    const calls = assistant.content.filter(part => part.type === "toolCall");
    assert.equal(calls.length, 2);
    assert.deepEqual(normalized.inferredToolCallIds, calls.map(call => call.id));
    assert.deepEqual(normalized.messages.slice(1).map(result => [result.toolCallId, result.content]), [
      [calls[1]!.id, "grep-result"], [calls[0]!.id, "read-result"],
    ]);
    assert.deepEqual(normalized.warnings, []);
  });

  it("keeps ambiguous same-name Gemini results without inventing an execution pairing", () => {
    const normalized = normalizeAnthropicGoogle({ contents: [
      { role: "model", parts: [
        { functionCall: { name: "read", args: { path: "a" } } },
        { functionCall: { name: "read", args: { path: "b" } } },
      ] },
      { role: "user", parts: [{ functionResponse: { name: "read", response: { output: "retained" } } }] },
    ] }, "google-vertex");
    assert.equal(normalized.messages[1]?.content, "retained");
    assert.equal(normalized.messages[1]?.toolCallId, undefined);
    assert.ok(normalized.warnings.some(warning => warning.includes("no unique preceding call")));
  });

  it("projects the Google SDK params envelope with config-level system, tools, thought signatures, and results", () => {
    const normalized = normalizeAnthropicGoogle(
      {
        model: "gemini-example",
        contents: [
          { role: "user", parts: [{ text: "Look this up" }, googleImagePart] },
          {
            role: "model",
            parts: [
              { text: "I am thinking.", thought: true, thoughtSignature: "dGhvdWdodA==" },
              {
                functionCall: { id: "call_123", name: "lookup", args: { query: "example" } },
                thoughtSignature: "Y2FsbA==",
              },
            ],
          },
          { role: "user", parts: [{ functionResponse: { id: "call_123", name: "lookup", response: { output: "A result" }, parts: [googleImagePart] } }] },
        ],
        config: {
          systemInstruction: { parts: [{ text: "Answer accurately." }] },
          tools: [
            {
              functionDeclarations: [
                {
                  name: "lookup",
                  description: "Look up a value",
                  parametersJsonSchema: { type: "object", properties: { query: { type: "string" } } },
                },
              ],
            },
          ],
          thinkingConfig: { includeThoughts: true },
        },
      },
      "google-generative-ai",
    );

    assert.deepEqual(normalized.systemPrompt, ["Answer accurately."]);
    assert.deepEqual(normalized.tools, [
      {
        name: "lookup",
        description: "Look up a value",
        parameters: { type: "object", properties: { query: { type: "string" } } },
      },
    ]);
    assert.deepEqual(normalized.messages, [
      {
        role: "user",
        content: [
          { type: "text", text: "Look this up" },
          { type: "image", mimeType: "image/jpeg", data: "amVwZw==" },
        ],
      },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "I am thinking." },
          { type: "toolCall", id: "call_123", name: "lookup", arguments: { query: "example" } },
        ],
      },
      {
        role: "toolResult",
        content: [
          { type: "text", text: "A result" },
          { type: "image", mimeType: "image/jpeg", data: "amVwZw==" },
        ],
        toolCallId: "call_123",
        toolName: "lookup",
      },
    ]);
    assert.equal(normalized.previousResponseId, undefined);
    assert.ok(normalized.warnings.some(warning => warning.includes("thoughtSignature")));
  });

  it("accepts flattened Vertex GenerateContent requests and inline data URIs", () => {
    const normalized = normalizeAnthropicGoogle(
      {
        model: "gemini-vertex-example",
        contents: [
          {
            role: "user",
            parts: [
              { fileData: { fileUri: "data:image/webp;base64,d2VicA==", mimeType: "image/webp" } },
              { text: "Describe this." },
            ],
          },
          { role: "model", parts: [{ text: "A webp image." }] },
        ],
        systemInstruction: { role: "user", parts: [{ text: "Be concise." }] },
        tools: [{ functionDeclarations: [{ name: "read", parameters: { type: "object", properties: {} } }] }],
        generationConfig: { maxOutputTokens: 250 },
      },
      "google-vertex",
    );

    assert.deepEqual(normalized.systemPrompt, ["Be concise."]);
    assert.deepEqual(normalized.tools, [{ name: "read", parameters: { type: "object", properties: {} } }]);
    assert.deepEqual(normalized.messages, [
      {
        role: "user",
        content: [
          { type: "image", mimeType: "image/webp", data: "d2VicA==" },
          { type: "text", text: "Describe this." },
        ],
      },
      { role: "assistant", content: [{ type: "text", text: "A webp image." }] },
    ]);
    assert.ok(normalized.warnings.some(warning => warning.includes("generationConfig")));
  });

  it("projects the Cloud Code Assist / Antigravity wrapper and prior execution ID", () => {
    const normalized = normalizeAnthropicGoogle(
      {
        project: "project-example",
        model: "gemini-antigravity-example",
        requestId: "agent/example/123/trajectory/2",
        requestType: "agent",
        userAgent: "antigravity",
        request: {
          sessionId: "-12345",
          contents: [
            { role: "user", parts: [{ text: "Continue" }] },
            { role: "model", parts: [{ functionCall: { id: "ag_call", name: "read", args: { path: "README" } } }] },
            { role: "user", parts: [{ functionResponse: { id: "ag_call", name: "read", response: { output: "contents" } } }] },
          ],
          systemInstruction: { role: "user", parts: [{ text: "Use tools when needed." }] },
          generationConfig: { maxOutputTokens: 512 },
          tools: [{ functionDeclarations: [{ name: "read", parameters: { type: "object" } }] }],
          toolConfig: { functionCallingConfig: { mode: "VALIDATED" } },
          labels: {
            last_execution_id: "response-previous",
            last_step_index: "1",
            model_enum: "GEMINI_MODEL",
            trajectory_id: "trajectory",
            used_claude: "false",
            used_claude_conservative: "false",
          },
        },
      },
      "google-gemini-cli",
    );

    assert.equal(normalized.previousResponseId, "response-previous");
    assert.deepEqual(normalized.systemPrompt, ["Use tools when needed."]);
    assert.deepEqual(normalized.messages, [
      { role: "user", content: [{ type: "text", text: "Continue" }] },
      { role: "assistant", content: [{ type: "toolCall", id: "ag_call", name: "read", arguments: { path: "README" } }] },
      { role: "toolResult", content: "contents", toolCallId: "ag_call", toolName: "read" },
    ]);
  });

  it("reports malformed and unsupported content without fetching remote files", () => {
    const normalized = normalizeAnthropicGoogle(
      {
        messages: [
          {
            role: "user",
            content: [
              { type: "image", source: { type: "url", url: "https://example.invalid/image.png" } },
              { type: "audio", source: "opaque" },
              17,
            ],
          },
        ],
      },
      "anthropic-messages",
    );
    assert.deepEqual(normalized.messages, [{ role: "user", content: [] }]);
    assert.ok(normalized.warnings.some(warning => warning.includes("not inline base64")));
    assert.ok(normalized.warnings.some(warning => warning.includes("Unsupported Anthropic content block audio")));
    assert.ok(normalized.warnings.some(warning => warning.includes("Invalid Anthropic content block")));


    const wrongEnvelope = normalizeAnthropicGoogle({ contents: [] }, "google-gemini-cli");
    assert.deepEqual(wrongEnvelope.messages, []);
    assert.ok(wrongEnvelope.warnings.some(warning => warning.includes("Missing required object $.request")));
    assert.ok(normalizeAnthropicGoogle({}, "unknown-api").warnings.some(warning => warning.includes("Unsupported provider API")));
  });
});
