import { describe, expect, it } from "vitest";
import type { Conversation } from "../src/message.js";
import { toAnthropic, toGemini, toOpenAI } from "../src/serialize.js";

/** Each serializer emits the correct wire shape and places system correctly. */

describe("toOpenAI", () => {
  it("puts system back into the array as the first message", () => {
    const conv: Conversation = {
      system: "Be terse.",
      messages: [{ role: "user", parts: [{ type: "text", text: "Hi" }] }],
    };
    const { output } = toOpenAI(conv);
    expect(output).toEqual({
      messages: [
        { role: "system", content: "Be terse." },
        { role: "user", content: "Hi" },
      ],
    });
  });
});

describe("toAnthropic", () => {
  it("emits top-level system and nests tool_result inside a user turn", () => {
    const conv: Conversation = {
      system: "Be terse.",
      messages: [
        {
          role: "assistant",
          parts: [{ type: "tool_call", id: "t1", name: "search", args: { q: "x" } }],
        },
        { role: "tool", parts: [{ type: "tool_result", id: "t1", content: "ok" }] },
      ],
    };
    const { output } = toAnthropic(conv);
    expect(output.system).toBe("Be terse.");
    expect(output.messages[0]).toEqual({
      role: "assistant",
      content: [{ type: "tool_use", id: "t1", name: "search", input: { q: "x" } }],
    });
    // tool result relocated into a user message
    expect(output.messages[1]).toEqual({
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "t1", content: "ok" }],
    });
  });
});

describe("toGemini", () => {
  it("uses the model role, inlineData images, and functionCall/functionResponse", () => {
    const conv: Conversation = {
      messages: [
        {
          role: "user",
          parts: [{ type: "image", mimeType: "image/png", data: "AAAA" }],
        },
        {
          role: "assistant",
          parts: [{ type: "tool_call", id: "t1", name: "search", args: { q: "x" } }],
        },
        {
          role: "tool",
          parts: [{ type: "tool_result", id: "t1", name: "search", content: '{"hits":2}' }],
        },
      ],
    };
    const { output } = toGemini(conv);
    expect(output.contents[0]).toEqual({
      role: "user",
      parts: [{ inlineData: { mimeType: "image/png", data: "AAAA" } }],
    });
    // assistant -> model
    expect(output.contents[1]).toEqual({
      role: "model",
      parts: [{ functionCall: { name: "search", args: { q: "x" } } }],
    });
    // functionResponse keyed by name, no id, response revived to an object
    expect(output.contents[2]).toEqual({
      role: "user",
      parts: [{ functionResponse: { name: "search", response: { hits: 2 } } }],
    });
  });
});
