import { describe, expect, it } from "vitest";
import { convert } from "../src/convert.js";

/**
 * A lossless conversation, expressed in each provider's format, must survive a
 * round trip (provider -> IR -> same provider) as an identity. If the identity
 * holds, the parser and serializer are genuine inverses on the happy path.
 */

describe("round trips are identity for lossless conversations", () => {
  it("openai -> openai", () => {
    const openai = {
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hi" },
        { role: "assistant", content: "Hello! How can I help?" },
        { role: "user", content: "What is 2+2?" },
      ],
    };
    const { output, notes } = convert("openai", "openai", openai);
    expect(output).toEqual(openai);
    expect(notes).toEqual([]);
  });

  it("anthropic -> anthropic", () => {
    const anthropic = {
      system: "You are helpful.",
      messages: [
        { role: "user", content: [{ type: "text", text: "Hi" }] },
        { role: "assistant", content: [{ type: "text", text: "Hello!" }] },
      ],
    };
    const { output } = convert("anthropic", "anthropic", anthropic);
    expect(output).toEqual(anthropic);
  });

  it("gemini -> gemini", () => {
    const gemini = {
      systemInstruction: { parts: [{ text: "You are helpful." }] },
      contents: [
        { role: "user", parts: [{ text: "Hi" }] },
        { role: "model", parts: [{ text: "Hello!" }] },
      ],
    };
    const { output } = convert("gemini", "gemini", gemini);
    expect(output).toEqual(gemini);
  });

  it("bedrock -> bedrock", () => {
    const bedrock = {
      system: [{ text: "You are helpful." }],
      messages: [
        { role: "user", content: [{ text: "Hi" }] },
        { role: "assistant", content: [{ text: "Hello!" }] },
      ],
    };
    const { output } = convert("bedrock", "bedrock", bedrock);
    expect(output).toEqual(bedrock);
  });
});
