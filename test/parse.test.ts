import { describe, expect, it } from "vitest";
import { isToolCall, isToolResult } from "../src/message.js";
import { parseAnthropic, parseGemini, parseOpenAI } from "../src/parse.js";

/** Each parser lifts its format into the canonical IR correctly. */

describe("parseOpenAI", () => {
  it("hoists a leading system message into Conversation.system", () => {
    const { conversation } = parseOpenAI({
      messages: [
        { role: "system", content: "Be terse." },
        { role: "user", content: "Hi" },
      ],
    });
    expect(conversation.system).toBe("Be terse.");
    expect(conversation.messages).toEqual([
      { role: "user", parts: [{ type: "text", text: "Hi" }] },
    ]);
  });

  it("lifts tool_calls and the tool role into IR ToolCall/ToolResult", () => {
    const { conversation } = parseOpenAI({
      messages: [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "get_weather", arguments: '{"city":"Paris"}' },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: "18C sunny" },
      ],
    });
    const call = conversation.messages[0]?.parts[0];
    const result = conversation.messages[1]?.parts[0];
    expect(call && isToolCall(call) ? call : null).toEqual({
      type: "tool_call",
      id: "call_1",
      name: "get_weather",
      args: { city: "Paris" },
    });
    expect(result && isToolResult(result) ? result : null).toEqual({
      type: "tool_result",
      id: "call_1",
      content: "18C sunny",
    });
  });
});

describe("parseAnthropic", () => {
  it("reads top-level system plus tool_use and tool_result blocks", () => {
    const { conversation } = parseAnthropic({
      system: "Be terse.",
      messages: [
        {
          role: "assistant",
          content: [{ type: "tool_use", id: "toolu_9", name: "search", input: { q: "cats" } }],
        },
        {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "toolu_9", content: "found" }],
        },
      ],
    });
    expect(conversation.system).toBe("Be terse.");
    expect(conversation.messages[0]).toEqual({
      role: "assistant",
      parts: [{ type: "tool_call", id: "toolu_9", name: "search", args: { q: "cats" } }],
    });
    // The tool_result block becomes its own IR `tool` message.
    expect(conversation.messages[1]).toEqual({
      role: "tool",
      parts: [{ type: "tool_result", id: "toolu_9", content: "found" }],
    });
  });
});

describe("parseGemini", () => {
  it("renames model->assistant and synthesizes tool-call ids by name/order", () => {
    const { conversation, collector } = parseGemini({
      contents: [
        { role: "model", parts: [{ functionCall: { name: "search", args: { q: "cats" } } }] },
        { role: "user", parts: [{ functionResponse: { name: "search", response: { hits: 3 } } }] },
      ],
    });
    // model -> assistant
    expect(conversation.messages[0]?.role).toBe("assistant");
    const call = conversation.messages[0]?.parts[0];
    const result = conversation.messages[1]?.parts[0];
    expect(call && isToolCall(call) ? call.id : null).toBe("gemini-call-0");
    // The response is paired to the call's synthesized id, by name/order.
    expect(result && isToolResult(result) ? result.id : null).toBe("gemini-call-0");
    expect(result && isToolResult(result) ? result.name : null).toBe("search");
    // Both the rename and the id synthesis are announced as info notes.
    const messages = collector.notes.map((n) => n.severity);
    expect(messages).toEqual(["info", "info"]);
  });
});
