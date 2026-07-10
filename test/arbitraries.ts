import fc from "fast-check";
import type { Provider } from "../src/convert.js";
import type { Conversation, ImagePart, Message, Part, TextPart, ToolCall } from "../src/message.js";

/**
 * fast-check generators for valid Conversation IRs.
 *
 * The IR is the pivot of the whole tool, so we generate it directly and feed it
 * through each serializer to produce provider payloads. Generating the IR (not a
 * flat Message[]) lets us enforce the one hard invariant at construction time:
 * every tool_result.id references a real tool_call.id, so the validator's
 * orphan-drop never fires in the lossless properties.
 *
 * Not a test file (no *.test.ts), so Vitest's `include` won't run it as a suite.
 */

// ── leaf arbitraries ─────────────────────────────────────────────────────────

const arbText = fc.string();
const arbNonEmptyText = fc.string({ minLength: 1 });
const arbMime = fc.constantFrom("image/png", "image/jpeg", "image/gif", "image/webp");
const arbBase64 = fc.base64String({ minLength: 0, maxLength: 32 });
const arbArgs = fc.jsonValue();
const arbToolName = fc.stringMatching(/^[a-zA-Z][a-zA-Z0-9_]{0,15}$/);

const arbTextPart = (allowEmptyText: boolean): fc.Arbitrary<TextPart> =>
  (allowEmptyText ? arbText : arbNonEmptyText).map((text) => ({ type: "text", text }));

const arbImagePart: fc.Arbitrary<ImagePart> = fc
  .record({ mimeType: arbMime, data: arbBase64 })
  .map(({ mimeType, data }) => ({ type: "image", mimeType, data }));

// ── turn descriptors ─────────────────────────────────────────────────────────

interface CallSpec {
  name: string;
  args: unknown;
}
type Turn =
  | { kind: "user"; parts: Part[] }
  | { kind: "assistant"; parts: Part[]; calls: CallSpec[] }
  | { kind: "answer"; n: number };

interface GenOpts {
  allowImages?: boolean;
  allowSystem?: boolean;
  allowEmptyText?: boolean;
}

function contentParts(opts: GenOpts): fc.Arbitrary<Part[]> {
  const part = opts.allowImages
    ? fc.oneof(arbTextPart(opts.allowEmptyText ?? false), arbImagePart)
    : arbTextPart(opts.allowEmptyText ?? false);
  return fc.array(part, { minLength: 1, maxLength: 3 });
}

const arbCallSpec: fc.Arbitrary<CallSpec> = fc
  .record({ name: arbToolName, args: arbArgs })
  .map(({ name, args }) => ({ name, args }));

function arbTurn(opts: GenOpts): fc.Arbitrary<Turn> {
  return fc.oneof(
    fc.record({ kind: fc.constant("user" as const), parts: contentParts(opts) }),
    fc.record({
      kind: fc.constant("assistant" as const),
      parts: contentParts(opts),
      calls: fc.array(arbCallSpec, { maxLength: 3 }),
    }),
    fc.record({ kind: fc.constant("answer" as const), n: fc.nat({ max: 3 }) }),
  );
}

/**
 * Lower a list of turns to a valid Message[]. Owns the id counter and the FIFO of
 * open (unanswered) tool calls, so every emitted tool_result answers a real call.
 */
function buildMessages(turns: Turn[]): Message[] {
  const messages: Message[] = [];
  const open: Array<{ id: string; name: string }> = [];
  let counter = 0;

  for (const turn of turns) {
    if (turn.kind === "user") {
      messages.push({ role: "user", parts: turn.parts });
    } else if (turn.kind === "assistant") {
      const calls: ToolCall[] = turn.calls.map((c) => {
        const id = `call_${counter++}`;
        open.push({ id, name: c.name });
        return { type: "tool_call", id, name: c.name, args: c.args };
      });
      messages.push({ role: "assistant", parts: [...turn.parts, ...calls] });
    } else {
      const count = Math.min(turn.n, open.length);
      for (let i = 0; i < count; i++) {
        const call = open.shift();
        if (call === undefined) break;
        messages.push({
          role: "tool",
          parts: [{ type: "tool_result", id: call.id, name: call.name, content: "ok" }],
        });
      }
    }
  }
  return messages;
}

// ── conversation factory, scoped per target ──────────────────────────────────

/** Drop tool results whose tool_call is not present (keeps the IR valid). */
function dropOrphanToolResults(messages: Message[]): Message[] {
  const callIds = new Set<string>();
  for (const m of messages) for (const p of m.parts) if (p.type === "tool_call") callIds.add(p.id);
  const out: Message[] = [];
  for (const m of messages) {
    const parts = m.parts.filter((p) => p.type !== "tool_result" || callIds.has(p.id));
    if (parts.length > 0) out.push({ ...m, parts });
  }
  return out;
}

/**
 * Ensure the message list is well-formed for Anthropic/Bedrock: drop leading
 * non-user turns and any message that would create a consecutive same-effective-
 * role run. Runs AFTER orphan removal, and re-runs orphan removal at the end so
 * a dropped assistant turn never leaves its tool answer orphaned.
 */
function wellFormedForStrictTarget(messages: Message[]): Message[] {
  const effectiveRole = (m: Message): "user" | "assistant" =>
    m.role === "assistant" ? "assistant" : "user";
  const out: Message[] = [];
  for (const m of messages) {
    if (out.length === 0 && effectiveRole(m) !== "user") continue;
    const last = out[out.length - 1];
    if (last !== undefined && effectiveRole(last) === effectiveRole(m)) continue;
    out.push(m);
  }
  // A dropped assistant turn may have minted a call whose answer is still here.
  return dropOrphanToolResults(out);
}

/**
 * Generate a Conversation the given target can represent LOSSLESSLY. Each target
 * excludes the things its serializer would reshape or drop:
 *  - anthropic/bedrock: no empty text, no leading non-user, no consecutive runs
 *  - gemini: tool ids are dropped by design (never asserted equal); no `extra`
 *  - openai: round-trips everything it parses
 */
export function arbConversationFor(target: Provider): fc.Arbitrary<Conversation> {
  const strict = target === "anthropic" || target === "bedrock";
  const opts: GenOpts = {
    allowImages: true,
    allowSystem: true,
    // Empty-string text parts are a degenerate case that no format round-trips as
    // meaningful content (an empty message has nothing to carry). They are
    // exercised explicitly in the seams tests, not treated as "lossless" here.
    allowEmptyText: false,
  };
  return fc
    .record({
      system: fc.option(arbNonEmptyText, { nil: undefined }),
      turns: fc.array(arbTurn(opts), { maxLength: 10 }),
    })
    .map(({ system, turns }) => {
      let messages = buildMessages(turns);
      if (strict) messages = wellFormedForStrictTarget(messages);
      return system === undefined ? { messages } : { system, messages };
    });
}

/** Count image parts across a whole conversation. */
export function countImages(conv: Conversation): number {
  let n = 0;
  for (const m of conv.messages) for (const p of m.parts) if (p.type === "image") n++;
  return n;
}

/** A generator over arbitrary JSON garbage, for the never-throws property. */
export const arbGarbage = fc.anything();
