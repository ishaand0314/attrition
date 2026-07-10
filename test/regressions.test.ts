import { describe, expect, it } from "vitest";
import { convert } from "../src/convert.js";
import type { LossNote } from "../src/notes.js";

/**
 * Regression tests for silent-corruption bugs the fuzz suite surfaced. These are
 * the specific inputs where a happy-path conversion looks fine but quietly
 * mutates content — the exact failure this tool exists to catch. Each bug is
 * pinned here so it can never come back without turning a test red.
 *
 * Bug #1 — Gemini pairs tool results to calls by function NAME, not id. Two open
 *          calls with the same name and out-of-order responses would mis-pair.
 * Bug #2 — JSON.parse ∘ JSON.stringify is not identity ("1.0" -> 1), so blindly
 *          reviving a tool-result string into a Gemini structured object mutates
 *          numbers silently.
 * Bug #4 — a remote image URL (mimeType "image/*") has no bytes to embed, so a
 *          base64-only target (Anthropic/Gemini/Bedrock) would write the URL
 *          string into a bytes field with no note.
 */

function find(notes: LossNote[], needle: string): LossNote | undefined {
  return notes.find((n) => n.message.toLowerCase().includes(needle.toLowerCase()));
}
function count(notes: LossNote[], needle: string): number {
  return notes.filter((n) => n.message.toLowerCase().includes(needle.toLowerCase())).length;
}

describe("bug #1: Gemini same-name tool-result pairing is ambiguous", () => {
  // Two functionCalls both named "f", then two functionResponses. Gemini pairs
  // by name + order, so if the responses are in a different order than the caller
  // intended, the tool can only pair by arrival order and cannot know it's wrong.
  // We can't fix ambiguity we have no information to resolve — but we MUST warn.
  const geminiPayload = {
    contents: [
      {
        role: "model",
        parts: [
          { functionCall: { name: "f", args: { which: 1 } } },
          { functionCall: { name: "f", args: { which: 2 } } },
        ],
      },
      {
        role: "user",
        parts: [{ functionResponse: { name: "f", response: { r: "second" } } }],
      },
    ],
  };

  it("emits a warning when >1 open call shares the response's name", () => {
    const { notes } = convert("gemini", "gemini", geminiPayload);
    const note = find(notes, "ambiguous gemini tool-result pairing");
    expect(note?.severity).toBe("warning");
  });

  it("does NOT warn when each open call has a distinct name", () => {
    const { notes } = convert("gemini", "gemini", {
      contents: [
        {
          role: "model",
          parts: [
            { functionCall: { name: "f", args: {} } },
            { functionCall: { name: "g", args: {} } },
          ],
        },
        { role: "user", parts: [{ functionResponse: { name: "f", response: { r: 1 } } }] },
      ],
    });
    expect(find(notes, "ambiguous gemini tool-result pairing")).toBeUndefined();
  });
});

describe("bug #2: tool-result JSON is not silently mutated on the way to Gemini", () => {
  // A Gemini-origin response object round-trips byte-exact via the extra stash,
  // even when it contains a value that JSON.parse/stringify would rewrite.
  it("gemini -> gemini revives a numeric response object exactly (1.0 stays 1.0)", () => {
    const payload = {
      contents: [
        { role: "model", parts: [{ functionCall: { name: "calc", args: {} } }] },
        {
          role: "user",
          parts: [{ functionResponse: { name: "calc", response: { value: 1.0, tag: "x" } } }],
        },
      ],
    };
    const { output } = convert("gemini", "gemini", payload) as {
      output: { contents: Array<{ parts: Array<Record<string, unknown>> }> };
    };
    const response = (output.contents[1]?.parts[0] as { functionResponse: { response: unknown } })
      .functionResponse.response;
    // The stash preserves the actual parsed object identity of the response.
    expect(response).toEqual({ value: 1.0, tag: "x" });
  });

  // A NON-Gemini-origin tool result is a plain string in the IR. We must not
  // JSON.parse it into a structured object when doing so would change bytes
  // (openai content "1.0" -> the number 1 loses the trailing zero silently).
  it("openai '1.0' tool result -> gemini does NOT become the number 1", () => {
    const payload = {
      messages: [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "calc", arguments: "{}" } },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: "1.0" },
      ],
    };
    const { output } = convert("openai", "gemini", payload) as {
      output: { contents: Array<{ parts: Array<Record<string, unknown>> }> };
    };
    // Find the functionResponse in the serialized Gemini payload.
    let response: unknown;
    for (const c of output.contents) {
      for (const p of c.parts) {
        if ("functionResponse" in p) {
          response = (p as { functionResponse: { response: unknown } }).functionResponse.response;
        }
      }
    }
    // It must NOT have been coerced to the number 1 (that would drop the ".0").
    expect(response).not.toBe(1);
    // The safe representation wraps the untouched string, so "1.0" survives verbatim.
    expect(response).toEqual({ result: "1.0" });
  });

  it("openai JSON-object tool result that re-stringifies exactly IS revived", () => {
    // {"ok":true} re-stringifies to the identical bytes, so parsing is lossless
    // and the structured object is the faithful representation.
    const payload = {
      messages: [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "calc", arguments: "{}" } },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: '{"ok":true}' },
      ],
    };
    const { output } = convert("openai", "gemini", payload) as {
      output: { contents: Array<{ parts: Array<Record<string, unknown>> }> };
    };
    let response: unknown;
    for (const c of output.contents) {
      for (const p of c.parts) {
        if ("functionResponse" in p) {
          response = (p as { functionResponse: { response: unknown } }).functionResponse.response;
        }
      }
    }
    expect(response).toEqual({ ok: true });
  });
});

describe("bug #4: a remote image URL is a loss on every base64-only target", () => {
  // OpenAI carries images as a URL; a remote (non-data:) URL has no bytes. The
  // OpenAI parser records it as mimeType "image/*" with the URL in `data`.
  const remoteImageOpenAI = {
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "look at this" },
          { type: "image_url", image_url: { url: "https://example.com/cat.png" } },
        ],
      },
    ],
  };

  it("openai -> anthropic drops the remote image with a loss note", () => {
    const { output, notes } = convert("openai", "anthropic", remoteImageOpenAI) as {
      output: { messages: Array<{ content: Array<Record<string, unknown>> }> };
      notes: LossNote[];
    };
    expect(find(notes, "remote image url")?.severity).toBe("loss");
    // The URL must NOT have been embedded as base64 bytes anywhere.
    const hasImageBlock = output.messages.some((m) => m.content.some((b) => b.type === "image"));
    expect(hasImageBlock).toBe(false);
  });

  it("openai -> gemini drops the remote image with a loss note", () => {
    const { output, notes } = convert("openai", "gemini", remoteImageOpenAI) as {
      output: { contents: Array<{ parts: Array<Record<string, unknown>> }> };
      notes: LossNote[];
    };
    expect(find(notes, "remote image url")?.severity).toBe("loss");
    const hasInlineData = output.contents.some((c) => c.parts.some((p) => "inlineData" in p));
    expect(hasInlineData).toBe(false);
  });

  it("openai -> bedrock drops the remote image with a loss note", () => {
    const { output, notes } = convert("openai", "bedrock", remoteImageOpenAI) as {
      output: { messages: Array<{ content: Array<Record<string, unknown>> }> };
      notes: LossNote[];
    };
    expect(find(notes, "remote image url")?.severity).toBe("loss");
    const hasImageBlock = output.messages.some((m) => m.content.some((b) => "image" in b));
    expect(hasImageBlock).toBe(false);
  });

  it("openai -> openai keeps the remote image URL (no loss, it can reference URLs)", () => {
    const { output, notes } = convert("openai", "openai", remoteImageOpenAI) as {
      output: { messages: Array<{ content: Array<Record<string, unknown>> }> };
      notes: LossNote[];
    };
    expect(find(notes, "remote image url")).toBeUndefined();
    expect(count(notes, "remote image url")).toBe(0);
    // The URL survives verbatim.
    const chunk = output.messages[0]?.content.find((b) => b.type === "image_url") as
      | { image_url: { url: string } }
      | undefined;
    expect(chunk?.image_url.url).toBe("https://example.com/cat.png");
  });
});
