import { describe, expect, it } from "vitest";
import { convert } from "../src/convert.js";
import type { LossNote } from "../src/notes.js";

/**
 * The 4th provider: AWS Bedrock Converse. These cover the seams no other format
 * hits (array-wrapped system, bare-enum image format, structured {json} tool
 * results, document blocks with no IR home) plus the Anthropic-style seams it
 * reuses for free (relocation, consecutive-merge, first-must-be-user).
 */

function find(notes: LossNote[], needle: string): LossNote | undefined {
  return notes.find((n) => n.message.toLowerCase().includes(needle.toLowerCase()));
}

describe("bedrock provider", () => {
  it("round-trips a structured {json} tool result, reviving the exact json block", () => {
    const payload = {
      messages: [
        {
          role: "assistant",
          content: [{ toolUse: { toolUseId: "tu_1", name: "get_weather", input: { city: "SF" } } }],
        },
        {
          role: "user",
          content: [
            {
              toolResult: {
                toolUseId: "tu_1",
                status: "success",
                content: [{ json: { tempF: 68 } }],
              },
            },
          ],
        },
      ],
    };
    const { output, notes } = convert("bedrock", "bedrock", payload) as {
      output: {
        messages: Array<{ content: Array<{ toolResult?: { content: unknown[] } }> }>;
      };
      notes: LossNote[];
    };
    // The structured content is announced...
    expect(find(notes, "structured (JSON)")?.severity).toBe("info");
    // ...and revived as {json:...}, NOT {text:"{...}"}.
    const tr = output.messages
      .flatMap((m) => m.content)
      .find((b) => b.toolResult !== undefined)?.toolResult;
    expect(tr?.content).toEqual([{ json: { tempF: 68 } }]);
  });

  it("drops an image whose MIME has no Bedrock enum, with a loss", () => {
    const { output, notes } = convert("openai", "bedrock", {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "look" },
            { type: "image_url", image_url: { url: "data:image/svg+xml;base64,PHN2Zz4=" } },
          ],
        },
      ],
    }) as { output: { messages: Array<{ content: unknown[] }> }; notes: LossNote[] };
    expect(find(notes, "unsupported Bedrock format")?.severity).toBe("loss");
    // The image is gone; only the text block survives.
    const blocks = output.messages.flatMap((m) => m.content);
    expect(blocks).toEqual([{ text: "look" }]);
  });

  it("drops a document block with a loss but still converts the surviving text", () => {
    const { output, notes } = convert("bedrock", "anthropic", {
      messages: [
        {
          role: "user",
          content: [
            { text: "read this" },
            { document: { name: "report.pdf", format: "pdf", source: { bytes: "JVBERi0=" } } },
          ],
        },
      ],
    }) as { output: { messages: Array<{ content: Array<{ type: string }> }> }; notes: LossNote[] };
    const losses = notes.filter((n) => n.severity === "loss");
    expect(losses).toHaveLength(1);
    expect(losses[0]?.message.toLowerCase()).toContain("document");
    // The text still made it across (loss is honest, not fatal).
    expect(output.messages[0]?.content).toEqual([{ type: "text", text: "read this" }]);
  });

  it("reuses the shared tool-result relocation into a valid Anthropic tool_result", () => {
    const { output, notes } = convert("bedrock", "anthropic", {
      messages: [
        {
          role: "assistant",
          content: [{ toolUse: { toolUseId: "tu_1", name: "s", input: {} } }],
        },
        {
          role: "user",
          content: [
            { toolResult: { toolUseId: "tu_1", status: "success", content: [{ text: "ok" }] } },
          ],
        },
      ],
    }) as {
      output: { messages: Array<{ role: string; content: Array<{ type: string }> }> };
      notes: LossNote[];
    };
    expect(find(notes, "relocated tool results")?.severity).toBe("info");
    const blockTypes = output.messages.flatMap((m) => m.content.map((b) => b.type));
    expect(blockTypes).toContain("tool_use");
    expect(blockTypes).toContain("tool_result");
  });

  it("array-wraps system and flags consecutive-user + leading-assistant like Anthropic", () => {
    const { output, notes } = convert("openai", "bedrock", {
      messages: [
        { role: "system", content: "Be terse." },
        { role: "assistant", content: "I'll start." },
        { role: "user", content: "First." },
        { role: "user", content: "Second." },
      ],
    }) as { output: { system?: Array<{ text: string }> }; notes: LossNote[] };
    // system is a [{text}] array, not a bare string.
    expect(output.system).toEqual([{ text: "Be terse." }]);
    expect(find(notes, "consecutive same-role")?.severity).toBe("warning");
    expect(find(notes, "start with a user turn")?.severity).toBe("warning");
  });
});
