import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { normalizeOpenAI } from "../src/backfill/openai.ts";

describe("normalizeOpenAI", () => {
  it("projects Chat Completions roles, tools, reasoning, images, calls, and results", () => {
    const parameters = {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
    };
    const normalized = normalizeOpenAI({
      messages: [
        { role: "system", content: "Use the available tools." },
        { role: "developer", content: "Return concise results." },
        {
          role: "user",
          content: [
            { type: "text", text: "Find the release notes." },
            { type: "image_url", image_url: { url: "data:image/png;base64,AA==" } },
          ],
        },
        {
          role: "assistant",
          reasoning_content: "I should search the documentation.",
          content: "Searching now.",
          tool_calls: [{
            id: "call_chat_17",
            type: "function",
            function: { name: "search_docs", arguments: '{"query":"release notes"}' },
          }],
        },
        { role: "tool", tool_call_id: "call_chat_17", name: "search_docs", content: "Release 4.2 shipped." },
      ],
      tools: [{ type: "function", function: { name: "search_docs", description: "Search documentation", parameters } }],
    }, "openai-completions");

    assert.deepEqual(normalized.systemPrompt, ["Use the available tools.", "Return concise results."]);
    assert.equal(normalized.delta, false);
    assert.deepEqual(normalized.tools, [{ name: "search_docs", description: "Search documentation", parameters }]);
    assert.deepEqual(normalized.messages, [
      {
        role: "user",
        content: [
          { type: "text", text: "Find the release notes." },
          { type: "image", mimeType: "image/png", data: "AA==" },
        ],
      },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "I should search the documentation." },
          { type: "text", text: "Searching now." },
          { type: "toolCall", id: "call_chat_17", name: "search_docs", arguments: { query: "release notes" } },
        ],
      },
      {
        role: "toolResult",
        toolCallId: "call_chat_17",
        toolName: "search_docs",
        content: "Release 4.2 shipped.",
      },
    ]);
    assert.deepEqual(normalized.warnings, []);
  });

  it("projects Responses instructions, reasoning, function calls, and matching outputs using wire call IDs", () => {
    const parameters = { type: "object", properties: { path: { type: "string" } } };
    const normalized = normalizeOpenAI({
      instructions: "Follow the repository policy.",
      input: [
        { type: "message", role: "developer", content: [{ type: "input_text", text: "Prefer small diffs." }] },
        {
          type: "message",
          role: "user",
          content: [
            { type: "input_text", text: "Read the changelog." },
            { type: "input_image", image_url: "data:image/jpeg;base64,/9j/2Q==" },
          ],
        },
        {
          type: "reasoning",
          summary: [{ type: "summary_text", text: "The request needs one file read." }],
          encrypted_content: "opaque-reasoning-token",
        },
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "I will read it." }] },
        {
          type: "function_call",
          id: "fc_item_9",
          call_id: "call_wire_9",
          name: "read_file",
          arguments: '{"path":"CHANGELOG.md"}',
        },
        { type: "function_call_output", call_id: "call_wire_9", output: [{ type: "input_text", text: "Version 4.2" }] },
      ],
      tools: [{ type: "function", name: "read_file", description: "Read a file", parameters }],
    }, "openai-responses");

    assert.deepEqual(normalized.systemPrompt, ["Follow the repository policy.", "Prefer small diffs."]);
    assert.equal(normalized.delta, false);
    assert.deepEqual(normalized.tools, [{ name: "read_file", description: "Read a file", parameters }]);
    assert.deepEqual(normalized.messages, [
      {
        role: "user",
        content: [
          { type: "text", text: "Read the changelog." },
          { type: "image", mimeType: "image/jpeg", data: "/9j/2Q==" },
        ],
      },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "The request needs one file read." },
          { type: "redactedThinking", data: "opaque-reasoning-token" },
        ],
      },
      { role: "assistant", content: [{ type: "text", text: "I will read it." }] },
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "call_wire_9", name: "read_file", arguments: { path: "CHANGELOG.md" } }],
      },
      {
        role: "toolResult",
        toolCallId: "call_wire_9",
        toolName: "read_file",
        content: [{ type: "text", text: "Version 4.2" }],
      },
    ]);
    assert.ok(normalized.warnings.some((message) => message.includes("fc_item_9") && message.includes("not represented")));
  });

  it("marks previous_response_id requests as deltas and keeps tool results separate from user input", () => {
    const normalized = normalizeOpenAI({
      type: "response.create",
      instructions: "Continue from the stored response.",
      previous_response_id: "resp_previous_42",
      input: [
        { type: "function_call_output", call_id: "call_from_previous_turn", output: "Task completed." },
        { type: "message", role: "user", content: [{ type: "input_text", text: "Now explain the result." }] },
      ],
    }, "openai-codex-responses");

    assert.deepEqual(normalized.systemPrompt, ["Continue from the stored response."]);
    assert.equal(normalized.delta, true);
    assert.equal(normalized.previousResponseId, "resp_previous_42");
    assert.deepEqual(normalized.messages, [
      { role: "toolResult", content: "Task completed.", toolCallId: "call_from_previous_turn" },
      { role: "user", content: [{ type: "text", text: "Now explain the result." }] },
    ]);
    assert.deepEqual(normalized.warnings, []);
  });

  it("warns for unsupported Responses items and non-replayable remote images instead of fabricating content", () => {
    const normalized = normalizeOpenAI({
      input: [
        { type: "web_search_call", id: "ws_call_1", status: "completed" },
        { type: "message", role: "user", content: [{ type: "input_image", image_url: "https://example.invalid/image.png" }] },
      ],
      tools: [{ type: "web_search_preview" }],
    }, "openai-responses");

    assert.deepEqual(normalized.messages, [{ role: "user", content: [] }]);
    assert.deepEqual(normalized.tools, []);
    assert.ok(normalized.warnings.some((message) => message.includes("web_search_call")));
    assert.ok(normalized.warnings.some((message) => message.includes("data URI") && message.includes("without IO")));
    assert.ok(normalized.warnings.some((message) => message.includes("web_search_preview") && message.includes("not representable")));
  });

  it("warns when Chat Completions audio cannot be represented", () => {
    const normalized = normalizeOpenAI({
      messages: [{ role: "assistant", audio: { id: "audio_response_1" } }],
    }, "openai-completions");

    assert.deepEqual(normalized.messages, []);
    assert.ok(normalized.warnings.some((message) => message.includes("audio") && message.includes("not representable")));
  });

  it("rejects unknown API names and malformed request envelopes", () => {
    assert.throws(() => normalizeOpenAI({}, "openai-unknown"), /Unsupported OpenAI API/);
    assert.throws(() => normalizeOpenAI({ messages: {} }, "openai-completions"), /messages array/);
    assert.throws(() => normalizeOpenAI({ input: {} }, "openai-responses"), /input to be a string or array/);
  });
});
