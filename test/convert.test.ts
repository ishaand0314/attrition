import { describe, expect, it } from "vitest";
import { convert } from "../src/convert.js";

/** End-to-end convert() across all three targets, including tool-call survival. */

describe("convert end to end", () => {
  it("a tool call + its matching result survive openai -> anthropic -> gemini", () => {
    const openai = {
      messages: [
        { role: "user", content: "Weather in Paris?" },
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
        { role: "tool", tool_call_id: "call_1", content: '{"temp":18}' },
        { role: "assistant", content: "It's 18C in Paris." },
      ],
    };

    // openai -> anthropic: tool_use + tool_result both present, no loss.
    const toAnthropic = convert("openai", "anthropic", openai) as {
      output: { messages: Array<{ role: string; content: Array<{ type: string }> }> };
      notes: Array<{ severity: string }>;
    };
    expect(toAnthropic.notes.some((n) => n.severity === "loss")).toBe(false);
    const blockTypes = toAnthropic.output.messages.flatMap((m) => m.content.map((b) => b.type));
    expect(blockTypes).toContain("tool_use");
    expect(blockTypes).toContain("tool_result");

    // openai -> gemini: functionCall + functionResponse both present, no loss.
    const toGemini = convert("openai", "gemini", openai) as {
      output: { contents: Array<{ parts: Array<Record<string, unknown>> }> };
      notes: Array<{ severity: string }>;
    };
    expect(toGemini.notes.some((n) => n.severity === "loss")).toBe(false);
    const partKeys = toGemini.output.contents.flatMap((c) =>
      c.parts.flatMap((p) => Object.keys(p)),
    );
    expect(partKeys).toContain("functionCall");
    expect(partKeys).toContain("functionResponse");
  });

  it("preserves the system prompt across every target", () => {
    const openai = {
      messages: [
        { role: "system", content: "You are a pirate." },
        { role: "user", content: "Hello" },
      ],
    };

    const anthropic = convert("openai", "anthropic", openai) as { output: { system?: string } };
    expect(anthropic.output.system).toBe("You are a pirate.");

    const gemini = convert("openai", "gemini", openai) as {
      output: { systemInstruction?: { parts: Array<{ text: string }> } };
    };
    expect(gemini.output.systemInstruction?.parts[0]?.text).toBe("You are a pirate.");
  });
});
