import { describe, expect, it } from "vitest";
import { convert } from "../src/convert.js";
import type { LossNote, Severity } from "../src/notes.js";

/**
 * The seams table IS the spec. Every row here is a place where information is
 * reshaped or lost between formats, and each produces an exact note.
 */

/** Find the first note whose message contains `needle` (case-insensitive). */
function find(notes: LossNote[], needle: string): LossNote | undefined {
  return notes.find((n) => n.message.toLowerCase().includes(needle.toLowerCase()));
}

function severitiesFor(notes: LossNote[], needle: string): Severity | undefined {
  return find(notes, needle)?.severity;
}

describe("seams", () => {
  it("multiple system messages -> anthropic merges with a warning", () => {
    const { output, notes } = convert("openai", "anthropic", {
      messages: [
        { role: "system", content: "One." },
        { role: "system", content: "Two." },
        { role: "user", content: "Hi" },
      ],
    }) as { output: { system?: string }; notes: LossNote[] };
    expect(output.system).toBe("One.\n\nTwo.");
    expect(severitiesFor(notes, "merged 2 system messages")).toBe("warning");
  });

  it("system not first (openai) -> hoisting it emits a warning", () => {
    const { notes } = convert("openai", "anthropic", {
      messages: [
        { role: "user", content: "Hi" },
        { role: "system", content: "Actually, be terse." },
      ],
    });
    expect(severitiesFor(notes, "appeared after other messages")).toBe("warning");
  });

  it("assistant role -> gemini renamed to model with an info (NOT a warning)", () => {
    const { notes } = convert("openai", "gemini", {
      messages: [{ role: "assistant", content: "Hello" }],
    });
    const note = find(notes, 'renamed role "assistant" to "model"');
    expect(note?.severity).toBe("info");
  });

  it("openai `name` field -> preserved in extra with an info", () => {
    const { notes } = convert("openai", "anthropic", {
      messages: [{ role: "user", name: "alice", content: "Hi" }],
    });
    const note = find(notes, "preserved openai `name`");
    expect(note?.severity).toBe("info");
  });

  it("orphan tool result (no matching call) -> dropped with exactly one loss note", () => {
    const { notes } = convert("openai", "openai", {
      messages: [{ role: "tool", tool_call_id: "ghost", content: "stray" }],
    });
    const losses = notes.filter((n) => n.severity === "loss");
    expect(losses).toHaveLength(1);
    expect(losses[0]?.message.toLowerCase()).toContain("no matching call");
  });

  it("orphan-drop that empties an anthropic user message -> additional empty-message warning", () => {
    // The orphan is the *only* part of its message; dropping it empties the
    // message, which triggers the second-order warning (seam H).
    const { notes } = convert("anthropic", "anthropic", {
      messages: [
        {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "ghost", content: "stray" }],
        },
      ],
    });
    expect(severitiesFor(notes, "no matching call")).toBe("loss");
    expect(severitiesFor(notes, "left empty after removing an orphan")).toBe("warning");
  });

  it("tool call with NO result survives all 3 targets and emits no loss (seam I)", () => {
    const payload = {
      messages: [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "c1", type: "function", function: { name: "search", arguments: "{}" } },
          ],
        },
      ],
    };
    for (const to of ["openai", "anthropic", "gemini"] as const) {
      const { notes } = convert("openai", to, payload);
      expect(notes.some((n) => n.severity === "loss")).toBe(false);
    }
  });

  it("image part with images allowed -> no loss (all three providers carry images)", () => {
    const payload = {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "look" },
            { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
          ],
        },
      ],
    };
    for (const to of ["openai", "anthropic", "gemini"] as const) {
      const { notes } = convert("openai", to, payload);
      expect(notes.some((n) => n.severity === "loss")).toBe(false);
    }
  });

  it("image part -> text-only target (--text-only) emits loss and drops the image", () => {
    const payload = {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "describe this" },
            { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
          ],
        },
      ],
    };
    // Every target model that can't take images gets the same honest loss.
    for (const to of ["openai", "anthropic", "gemini"] as const) {
      const { notes } = convert("openai", to, payload, { textOnly: true });
      const loss = find(notes, "text-only");
      expect(loss?.severity).toBe("loss");
    }
  });

  it("consecutive same-role messages -> anthropic merge with a warning", () => {
    const { output, notes } = convert("openai", "anthropic", {
      messages: [
        { role: "user", content: "First." },
        { role: "user", content: "Second." },
      ],
    }) as { output: { messages: unknown[] }; notes: LossNote[] };
    expect(output.messages).toHaveLength(1);
    expect(severitiesFor(notes, "consecutive same-role")).toBe("warning");
  });

  it("empty content block -> anthropic drop with a warning", () => {
    const { notes } = convert("openai", "anthropic", {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "" },
            { type: "text", text: "real" },
          ],
        },
      ],
    });
    expect(severitiesFor(notes, "empty text block")).toBe("warning");
  });

  it("leading assistant turn -> anthropic first-must-be-user warning (seam B)", () => {
    const { notes } = convert("openai", "anthropic", {
      messages: [{ role: "assistant", content: "I'll start." }],
    });
    expect(severitiesFor(notes, "start with a user turn")).toBe("warning");
  });

  it("going to gemini drops tool ids with an info (seam A)", () => {
    const { notes } = convert("anthropic", "gemini", {
      messages: [
        { role: "assistant", content: [{ type: "tool_use", id: "toolu_x", name: "s", input: {} }] },
      ],
    });
    expect(severitiesFor(notes, "dropped tool-call ids for gemini")).toBe("info");
  });

  it("system-only conversation -> anthropic empty-messages warning (seam F)", () => {
    const { output, notes } = convert("openai", "anthropic", {
      messages: [{ role: "system", content: "Only a system prompt." }],
    }) as { output: { system?: string; messages: unknown[] }; notes: LossNote[] };
    expect(output.system).toBe("Only a system prompt.");
    expect(output.messages).toEqual([]);
    expect(severitiesFor(notes, "requires a non-empty messages array")).toBe("warning");
  });
});
