import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { PROVIDERS, type Provider, convert } from "../src/convert.js";
import { type LossNote, hasLoss } from "../src/notes.js";
import { arbConversationFor, arbGarbage, countImages } from "./arbitraries.js";

/**
 * Property-based tests. These generate random valid IR conversations, serialize
 * them to a provider payload, and assert the translator's invariants hold for
 * ALL of them — catching the silent-corruption bugs the happy-path suite can't.
 *
 * A serializer is reached via convert(p, p, payload): parse -> validate ->
 * serialize. We build the IR directly (arbitraries), so the "source" payload is
 * S_p(conv) and we assert on the p->p (or p->q) convert of it.
 */

/** Serialize a generated IR conversation to a provider payload. */
function serialize(target: Provider, conv: Parameters<typeof countImages>[0]): unknown {
  // convert(target, target, ...) round-trips; but we need S_target(conv) first.
  // The public surface only exposes convert(from,to,payload); to serialize an IR
  // directly we go through the barrel's serializers via a from==to identity on a
  // payload we build by hand is circular. Instead use the exported serializers.
  return serializeIr(target, conv);
}

// Import serializers directly to turn an IR into a payload (the generators
// produce IR, not provider payloads).
import { toAnthropic, toBedrock, toGemini, toOpenAI } from "../src/serialize.js";
function serializeIr(target: Provider, conv: Parameters<typeof countImages>[0]): unknown {
  switch (target) {
    case "openai":
      return toOpenAI(conv).output;
    case "anthropic":
      return toAnthropic(conv).output;
    case "gemini":
      return toGemini(conv).output;
    case "bedrock":
      return toBedrock(conv).output;
  }
}

function losses(notes: LossNote[]): LossNote[] {
  return notes.filter((n) => n.severity === "loss");
}

describe("properties: no-throw and strict correctness (hold for ALL input)", () => {
  it("convert never throws for any (from, to, garbage, opts)", () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...PROVIDERS),
        fc.constantFrom(...PROVIDERS),
        arbGarbage,
        fc.boolean(),
        (from, to, payload, textOnly) => {
          expect(() => convert(from, to, payload, { textOnly })).not.toThrow();
        },
      ),
      { numRuns: 400 },
    );
  });

  it("hasLoss(notes) iff a loss note exists (independent recompute)", () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...PROVIDERS),
        fc.constantFrom(...PROVIDERS),
        arbGarbage,
        (from, to, payload) => {
          const { notes } = convert(from, to, payload);
          expect(hasLoss(notes)).toBe(notes.some((n) => n.severity === "loss"));
        },
      ),
      { numRuns: 300 },
    );
  });

  it("convert(p,p, convert(p,p,x).output) is a fixed point for garbage input", () => {
    fc.assert(
      fc.property(fc.constantFrom(...PROVIDERS), arbGarbage, (p, payload) => {
        const once = convert(p, p, payload).output;
        const twice = convert(p, p, once).output;
        expect(twice).toEqual(once);
      }),
      { numRuns: 300 },
    );
  });
});

describe("properties: image -> text-only is always a loss", () => {
  it("textOnly drops exactly as many images as the conversation had, per target", () => {
    fc.assert(
      fc.property(fc.constantFrom(...PROVIDERS), (target) => {
        return fc.assert(
          fc.property(arbConversationFor(target), (conv) => {
            const k = countImages(conv);
            const payload = serialize(target, conv);
            const { notes } = convert(target, target, payload, { textOnly: true });
            // Every image becomes a loss (at least k; parsers may add none).
            expect(losses(notes).filter((n) => n.message.includes("text-only")).length).toBe(k);
          }),
          { numRuns: 40 },
        );
      }),
      { numRuns: 1 },
    );
  });
});

describe("properties: lossless round trips and system preservation", () => {
  for (const target of PROVIDERS) {
    it(`${target} -> ${target} identity emits no loss on the lossless subset`, () => {
      fc.assert(
        fc.property(arbConversationFor(target), (conv) => {
          const payload = serialize(target, conv);
          const { notes } = convert(target, target, payload);
          expect(losses(notes)).toEqual([]);
        }),
        { numRuns: 120 },
      );
    });

    it(`${target} -> ${target} is a serialize fixed point (S∘P∘S == S)`, () => {
      fc.assert(
        fc.property(arbConversationFor(target), (conv) => {
          const payload = serialize(target, conv);
          const roundTripped = convert(target, target, payload).output;
          expect(roundTripped).toEqual(payload);
        }),
        { numRuns: 120 },
      );
    });
  }

  it("system prompt is preserved byte-identical across every target", () => {
    fc.assert(
      fc.property(fc.constantFrom(...PROVIDERS), fc.constantFrom(...PROVIDERS), (from, to) => {
        return fc.assert(
          fc.property(
            arbConversationFor(from).filter((c) => c.system !== undefined),
            (conv) => {
              const payload = serialize(from, conv);
              const { output } = convert(from, to, payload);
              expect(extractSystem(to, output)).toBe(conv.system);
            },
          ),
          { numRuns: 30 },
        );
      }),
      { numRuns: 1 },
    );
  });
});

function extractSystem(target: Provider, output: unknown): string | undefined {
  const o = output as Record<string, unknown>;
  if (target === "anthropic") {
    return typeof o.system === "string" ? o.system : undefined;
  }
  if (target === "bedrock") {
    const sys = o.system;
    if (!Array.isArray(sys)) return undefined;
    return sys
      .map((b) =>
        typeof (b as Record<string, unknown>).text === "string" ? (b as { text: string }).text : "",
      )
      .join("\n\n");
  }
  if (target === "gemini") {
    const si = o.systemInstruction as Record<string, unknown> | undefined;
    const parts = si?.parts;
    if (!Array.isArray(parts)) return undefined;
    return parts.map((p) => (p as { text: string }).text).join("\n\n");
  }
  // openai: first system message content
  const messages = o.messages;
  if (!Array.isArray(messages)) return undefined;
  const first = messages[0] as Record<string, unknown> | undefined;
  if (first?.role === "system" && typeof first.content === "string") return first.content;
  return undefined;
}
