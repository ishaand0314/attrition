/**
 * The canonical message IR.
 *
 * ── HAND-SYNCED WITH LABKIT ──────────────────────────────────────────────
 * This type is the same canonical `Message` shape used by labkit
 * (@labkit/token-cost's monorepo). There is NO shared npm package between the
 * two repos — this file is copied and kept in sync BY HAND. If you change the
 * shape here, mirror it in labkit, and vice versa. Do not let the two diverge
 * silently.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Everything a conversation means, independent of any provider's wire format.
 * Parsers lift a provider payload into this shape; serializers lower it back
 * out. The whole point of the IR is that adding a provider is one parser + one
 * serializer — never a new pairwise converter.
 */

export type Role = "system" | "user" | "assistant" | "tool";

export interface TextPart {
  type: "text";
  text: string;
}

/** An inline image. `data` is base64; `mimeType` e.g. "image/png". */
export interface ImagePart {
  type: "image";
  mimeType: string;
  data: string;
}

/** A request by the assistant to call a tool. `args` is arbitrary JSON. */
export interface ToolCall {
  type: "tool_call";
  id: string;
  name: string;
  args: unknown;
}

/** The result of a tool call, answering the `ToolCall` with the same `id`. */
export interface ToolResult {
  type: "tool_result";
  /** The id of the ToolCall this answers. */
  id: string;
  /**
   * The tool/function name this result is for. Gemini keys tool results by
   * function name rather than by id, so this is preserved for that serializer
   * and synthesized by the Gemini parser (which has no id to read).
   */
  name?: string;
  content: string;
  isError?: boolean;
}

export type Part = TextPart | ImagePart | ToolCall | ToolResult;

export interface Message {
  role: Role;
  parts: Part[];
  /**
   * Provider-specific fields that were parsed but cannot be represented in the
   * IR, kept for round-trip fidelity (e.g. OpenAI's `name`, or the original
   * vendor tool-call id when it was remapped).
   *
   * Keyed by "<provider>.<field>" (e.g. "openai.name") so a serializer can find
   * its own leftovers without colliding with another provider's on a multi-hop
   * conversion.
   */
  extra?: Record<string, unknown>;
}

export interface Conversation {
  /**
   * The system prompt, hoisted out of the message array. OpenAI carries it as
   * an in-array message; Anthropic and Gemini take it as a top-level field. The
   * IR holds the meaning; each serializer decides where it physically goes.
   */
  system?: string;
  messages: Message[];
}

// ── Part type guards (used by validate/serialize) ────────────────────────────

export function isText(part: Part): part is TextPart {
  return part.type === "text";
}

export function isImage(part: Part): part is ImagePart {
  return part.type === "image";
}

export function isToolCall(part: Part): part is ToolCall {
  return part.type === "tool_call";
}

export function isToolResult(part: Part): part is ToolResult {
  return part.type === "tool_result";
}
