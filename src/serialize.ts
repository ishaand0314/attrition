/**
 * Serializers: the canonical `Conversation` IR -> a provider's wire payload.
 *
 * One serializer per provider. Each decides where the hoisted system prompt
 * physically goes, how to represent tool calls/results, and flags every seam
 * where information is reshaped or lost via `LossNote`s.
 *
 * Serializers assume the IR is already valid (validate.ts ran): no orphan tool
 * results, no empty messages from orphan drops. They still handle format-
 * specific reshaping (consecutive-role merges, empty-block drops, images into a
 * text-only target).
 *
 * Every serializer returns `ConvertResult<T>`: the output plus its notes.
 */

import {
  type Conversation,
  type ImagePart,
  type Message,
  type Part,
  type Role,
  type TextPart,
  type ToolCall,
  type ToolResult,
  isImage,
  isText,
  isToolCall,
  isToolResult,
} from "./message.js";
import { type ConvertResult, type LossNote, NoteCollector } from "./notes.js";

/**
 * Serializers accept the pre-collected notes from parse + validate so the final
 * `ConvertResult.notes` is the full story of the conversion, in order.
 */
type Priors = readonly LossNote[];

/**
 * Per-serialization options.
 *
 * All three providers can carry images, so image loss only happens when the
 * caller declares the *target model* is text-only (many are). `textOnly: true`
 * drops image parts and records a `loss` — this is the "image into a text-only
 * target" seam. It is a property of the destination model, which the format
 * name alone doesn't tell us, so the caller supplies it.
 */
export interface SerializeOptions {
  textOnly?: boolean;
}

// ── OpenAI ───────────────────────────────────────────────────────────────────
//
// system goes back into the array as the first message. Tool results are their
// own `role: "tool"` messages. Assistant tool calls attach to the assistant
// message's `tool_calls`. Content is a plain string when it's a single text
// part, else an array of typed chunks (OpenAI supports both).

interface OpenAIMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | OpenAIContentChunk[] | null;
  name?: string;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}
type OpenAIContentChunk =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };
interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}
interface OpenAIPayload {
  messages: OpenAIMessage[];
}

export function toOpenAI(
  conversation: Conversation,
  priors: Priors = [],
  options: SerializeOptions = {},
): ConvertResult<OpenAIPayload> {
  const notes = new NoteCollector();
  const messages: OpenAIMessage[] = [];

  if (conversation.system !== undefined) {
    messages.push({ role: "system", content: conversation.system });
  }

  conversation.messages.forEach((msg, mi) => {
    if (msg.role === "tool") {
      for (const part of msg.parts) {
        if (isToolResult(part)) {
          messages.push({ role: "tool", tool_call_id: part.id, content: part.content });
        }
      }
      return;
    }

    const content: OpenAIContentChunk[] = [];
    const toolCalls: OpenAIToolCall[] = [];
    msg.parts.forEach((part, pi) => {
      if (isText(part)) {
        content.push({ type: "text", text: part.text });
      } else if (isImage(part)) {
        if (options.textOnly) {
          notes.loss(
            "Dropped an image part (target model is text-only)",
            `messages[${mi}].parts[${pi}]`,
          );
        } else {
          content.push({ type: "image_url", image_url: { url: imageToDataUrl(part) } });
        }
      } else if (isToolCall(part)) {
        toolCalls.push({
          id: part.id,
          type: "function",
          function: { name: part.name, arguments: stringifyArgs(part.args) },
        });
      }
    });

    const message: OpenAIMessage = {
      role: openaiRole(msg.role),
      content: collapseContent(content),
    };
    if (toolCalls.length > 0) message.tool_calls = toolCalls;
    const name = readExtraString(msg, "openai.name");
    if (name !== undefined) message.name = name;
    messages.push(message);
  });

  return result({ messages }, priors, notes);
}

function openaiRole(role: Role): "user" | "assistant" {
  return role === "assistant" ? "assistant" : "user";
}

/** A single text part serializes as a plain string (so round-trips are exact). */
function collapseContent(chunks: OpenAIContentChunk[]): string | OpenAIContentChunk[] | null {
  if (chunks.length === 0) return null;
  const first = chunks[0];
  if (chunks.length === 1 && first !== undefined && first.type === "text") {
    return first.text;
  }
  return chunks;
}

function imageToDataUrl(part: ImagePart): string {
  // A non-data image (mimeType "image/*") was originally a remote URL; keep it.
  if (isRemoteImage(part)) return part.data;
  return `data:${part.mimeType};base64,${part.data}`;
}

/**
 * True when an ImagePart carries a remote URL rather than base64 bytes. The
 * OpenAI parser records a remote `image_url` as `mimeType: "image/*"` with the
 * URL in `data` (there are no bytes to embed). OpenAI can re-emit that URL, but
 * every base64-only target (Anthropic, Gemini, Bedrock) would embed the URL
 * string into a bytes field — silent corruption — so those serializers drop it
 * with a `loss` instead. See `remoteImageLoss`.
 */
function isRemoteImage(part: ImagePart): boolean {
  return part.mimeType === "image/*";
}

const REMOTE_IMAGE_LOSS =
  "Dropped a remote image URL (this target embeds images as base64 bytes and cannot reference a URL)";

// ── Anthropic ─────────────────────────────────────────────────────────────────
//
// system is a top-level field. Messages alternate user/assistant; consecutive
// same-role messages are merged (Anthropic rejects them). Tool results nest as
// tool_result blocks INSIDE a user message. First message must be `user`.

interface AnthropicPayload {
  system?: string;
  messages: AnthropicMessage[];
}
interface AnthropicMessage {
  role: "user" | "assistant";
  content: AnthropicBlock[];
}
type AnthropicBlock =
  | { type: "text"; text: string }
  | { type: "image"; source: { type: "base64"; media_type: string; data: string } }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean };

export function toAnthropic(
  conversation: Conversation,
  priors: Priors = [],
  options: SerializeOptions = {},
): ConvertResult<AnthropicPayload> {
  const notes = new NoteCollector();

  // Map each IR message to an { role, blocks }, relocating tool results into a
  // user role, then merge consecutive same-role runs.
  interface Pending {
    role: "user" | "assistant";
    blocks: AnthropicBlock[];
  }
  const pending: Pending[] = [];
  let relocatedTool = false;

  conversation.messages.forEach((msg, mi) => {
    const role: "user" | "assistant" = msg.role === "assistant" ? "assistant" : "user";
    if (msg.role === "tool") relocatedTool = true;
    const blocks = anthropicBlocks(msg.parts, notes, options, mi);
    if (blocks.length === 0) return; // empty after dropping (see anthropicBlocks)
    pending.push({ role, blocks });
  });

  if (relocatedTool) {
    notes.info(
      'Relocated tool results into a "user" message (Anthropic nests tool_result in user turns)',
    );
  }

  // Merge consecutive same-role messages.
  const merged: AnthropicMessage[] = [];
  let mergeCount = 0;
  for (const p of pending) {
    const last = merged[merged.length - 1];
    if (last !== undefined && last.role === p.role) {
      last.content.push(...p.blocks);
      mergeCount++;
    } else {
      merged.push({ role: p.role, content: p.blocks });
    }
  }
  if (mergeCount > 0) {
    notes.warning(`Merged ${mergeCount} consecutive same-role message(s) (Anthropic rejects them)`);
  }

  // Anthropic requires the first message to be `user`.
  if (merged.length === 0) {
    notes.warning("Conversation has no messages; Anthropic requires a non-empty messages array");
  } else if (merged[0]?.role !== "user") {
    notes.warning(
      "First message is not `user`; Anthropic requires the conversation to start with a user turn",
    );
  }

  const payload: AnthropicPayload =
    conversation.system === undefined
      ? { messages: merged }
      : { system: conversation.system, messages: merged };
  return result(payload, priors, notes);
}

function anthropicBlocks(
  parts: Part[],
  notes: NoteCollector,
  options: SerializeOptions,
  mi: number,
): AnthropicBlock[] {
  const blocks: AnthropicBlock[] = [];
  parts.forEach((part, pi) => {
    if (isText(part)) {
      if (part.text.length === 0) {
        notes.warning(
          "Dropped an empty text block (Anthropic rejects empty content)",
          `messages[${mi}].parts[${pi}]`,
        );
        return;
      }
      blocks.push({ type: "text", text: part.text });
    } else if (isImage(part)) {
      if (options.textOnly) {
        notes.loss(
          "Dropped an image part (target model is text-only)",
          `messages[${mi}].parts[${pi}]`,
        );
        return;
      }
      if (isRemoteImage(part)) {
        notes.loss(REMOTE_IMAGE_LOSS, `messages[${mi}].parts[${pi}]`);
        return;
      }
      blocks.push({
        type: "image",
        source: { type: "base64", media_type: part.mimeType, data: part.data },
      });
    } else if (isToolCall(part)) {
      blocks.push({ type: "tool_use", id: part.id, name: part.name, input: part.args });
    } else if (isToolResult(part)) {
      const block: AnthropicBlock = {
        type: "tool_result",
        tool_use_id: part.id,
        content: part.content,
      };
      if (part.isError === true) block.is_error = true;
      blocks.push(block);
    }
  });
  return blocks;
}

// ── Gemini ────────────────────────────────────────────────────────────────────
//
// systemInstruction is top-level. role "assistant" -> "model". Images use
// inlineData. Tool calls/results use functionCall/functionResponse and carry NO
// id — matched by name — so the id is dropped (kept only in the IR).

interface GeminiPayload {
  systemInstruction?: { parts: Array<{ text: string }> };
  contents: GeminiContent[];
}
interface GeminiContent {
  role: "user" | "model";
  parts: GeminiPart[];
}
type GeminiPart =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } }
  | { functionCall: { name: string; args: unknown } }
  | { functionResponse: { name: string; response: unknown } };

export function toGemini(
  conversation: Conversation,
  priors: Priors = [],
  options: SerializeOptions = {},
): ConvertResult<GeminiPayload> {
  const notes = new NoteCollector();
  const contents: GeminiContent[] = [];
  let renamedAssistant = false;
  let droppedToolId = false;

  conversation.messages.forEach((msg, mi) => {
    const parts: GeminiPart[] = [];
    msg.parts.forEach((part, pi) => {
      if (isText(part)) {
        parts.push({ text: part.text });
      } else if (isImage(part)) {
        if (options.textOnly) {
          notes.loss(
            "Dropped an image part (target model is text-only)",
            `messages[${mi}].parts[${pi}]`,
          );
        } else if (isRemoteImage(part)) {
          notes.loss(REMOTE_IMAGE_LOSS, `messages[${mi}].parts[${pi}]`);
        } else {
          parts.push({ inlineData: { mimeType: part.mimeType, data: part.data } });
        }
      } else if (isToolCall(part)) {
        droppedToolId = true;
        parts.push({ functionCall: { name: part.name, args: part.args } });
      } else if (isToolResult(part)) {
        droppedToolId = true;
        parts.push({
          functionResponse: {
            name: part.name ?? part.id,
            response: geminiResponse(part, msg),
          },
        });
      }
    });
    if (parts.length === 0) return;

    let role: "user" | "model";
    if (msg.role === "assistant") {
      role = "model";
      renamedAssistant = true;
    } else {
      // user AND tool both map to "user" (Gemini nests functionResponse in a user turn).
      role = "user";
    }
    contents.push({ role, parts });
  });

  if (renamedAssistant) {
    notes.info('Renamed role "assistant" to "model" for Gemini');
  }
  if (droppedToolId) {
    notes.info("Dropped tool-call ids for Gemini (matches calls to responses by function name)");
  }

  const payload: GeminiPayload =
    conversation.system === undefined
      ? { contents }
      : { systemInstruction: { parts: [{ text: conversation.system }] }, contents };
  return result(payload, priors, notes);
}

function geminiResponse(part: ToolResult, msg: Message): unknown {
  // Gemini functionResponse.response is a structured object. For a result that
  // came from Gemini originally, revive the exact object we stashed at parse time
  // (JSON.stringify∘JSON.parse is not identity, so we must not re-derive it).
  const stashed = msg.extra?.[`gemini.functionResponse.${part.id}`];
  if (stashed !== undefined) return stashed;
  // For a result from any other format, the IR content is a plain string. Parse it
  // to a structured object ONLY when doing so is lossless (re-stringifying yields
  // the identical bytes) — otherwise "1.0"->1 or key reordering would silently
  // mutate it. When it isn't safe, wrap as { result: <string> } with no mutation.
  try {
    const parsed: unknown = JSON.parse(part.content);
    if (typeof parsed === "object" && parsed !== null && JSON.stringify(parsed) === part.content) {
      return parsed;
    }
  } catch {
    // not JSON; fall through to the safe wrapper
  }
  return { result: part.content };
}

// ── Bedrock (AWS Bedrock Converse API) ────────────────────────────────────────
//
// system is a top-level ARRAY of { text } blocks. Roles are user | assistant
// (assistant/user already match the IR — no rename). Tool results nest in a user
// turn and consecutive same-role messages are merged, exactly like Anthropic.
// Images use a bare-enum `format` (png|jpeg|gif|webp), so images whose MIME has
// no Bedrock enum are dropped as a `loss`. Tool results are an array that can
// carry structured { json } content.

interface BedrockPayload {
  system?: Array<{ text: string }>;
  messages: BedrockMessage[];
}
interface BedrockMessage {
  role: "user" | "assistant";
  content: BedrockBlock[];
}
type BedrockBlock =
  | { text: string }
  | { image: { format: string; source: { bytes: string } } }
  | { document: unknown }
  | { toolUse: { toolUseId: string; name: string; input: unknown } }
  | {
      toolResult: {
        toolUseId: string;
        status: "success" | "error";
        content: BedrockToolResultContent[];
      };
    };
type BedrockToolResultContent = { text: string } | { json: unknown };

const MIME_TO_BEDROCK_IMAGE_FORMAT: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpeg",
  "image/gif": "gif",
  "image/webp": "webp",
};

export function toBedrock(
  conversation: Conversation,
  priors: Priors = [],
  options: SerializeOptions = {},
): ConvertResult<BedrockPayload> {
  const notes = new NoteCollector();

  interface Pending {
    role: "user" | "assistant";
    blocks: BedrockBlock[];
  }
  const pending: Pending[] = [];
  let relocatedTool = false;
  let restoredDocs = 0;

  conversation.messages.forEach((msg, mi) => {
    const role: "user" | "assistant" = msg.role === "assistant" ? "assistant" : "user";
    if (msg.role === "tool") relocatedTool = true;
    const blocks = bedrockBlocks(msg, notes, options, mi);
    // Restore any documents stashed on this message from a prior bedrock parse.
    const docs = msg.extra?.["bedrock.documents"];
    if (Array.isArray(docs)) {
      for (const doc of docs) blocks.push({ document: asDocumentPayload(doc) });
      restoredDocs += docs.length;
    }
    if (blocks.length === 0) return;
    pending.push({ role, blocks });
  });

  if (relocatedTool) {
    notes.info(
      'Relocated tool results into a "user" message (Bedrock nests toolResult in user turns)',
    );
  }
  if (restoredDocs > 0) {
    notes.info(`Restored ${restoredDocs} Bedrock document block(s) from extra`);
  }

  // Merge consecutive same-role messages (Bedrock, like Anthropic, rejects them).
  const merged: BedrockMessage[] = [];
  let mergeCount = 0;
  for (const p of pending) {
    const last = merged[merged.length - 1];
    if (last !== undefined && last.role === p.role) {
      last.content.push(...p.blocks);
      mergeCount++;
    } else {
      merged.push({ role: p.role, content: p.blocks });
    }
  }
  if (mergeCount > 0) {
    notes.warning(`Merged ${mergeCount} consecutive same-role message(s) (Bedrock rejects them)`);
  }

  if (merged.length === 0) {
    notes.warning("Conversation has no messages; Bedrock requires a non-empty messages array");
  } else if (merged[0]?.role !== "user") {
    notes.warning(
      "First message is not `user`; Bedrock requires the conversation to start with a user turn",
    );
  }

  const payload: BedrockPayload =
    conversation.system === undefined
      ? { messages: merged }
      : { system: [{ text: conversation.system }], messages: merged };
  return result(payload, priors, notes);
}

function bedrockBlocks(
  msg: Message,
  notes: NoteCollector,
  options: SerializeOptions,
  mi: number,
): BedrockBlock[] {
  const blocks: BedrockBlock[] = [];
  msg.parts.forEach((part, pi) => {
    if (isText(part)) {
      if (part.text.length === 0) {
        notes.warning(
          "Dropped an empty text block (Bedrock rejects empty content)",
          `messages[${mi}].parts[${pi}]`,
        );
        return;
      }
      blocks.push({ text: part.text });
    } else if (isImage(part)) {
      if (options.textOnly) {
        notes.loss(
          "Dropped an image part (target model is text-only)",
          `messages[${mi}].parts[${pi}]`,
        );
        return;
      }
      if (isRemoteImage(part)) {
        notes.loss(REMOTE_IMAGE_LOSS, `messages[${mi}].parts[${pi}]`);
        return;
      }
      const format = MIME_TO_BEDROCK_IMAGE_FORMAT[part.mimeType];
      if (format === undefined) {
        notes.loss(
          `Dropped an image with unsupported Bedrock format (mimeType: "${part.mimeType}"); Bedrock allows only png/jpeg/gif/webp`,
          `messages[${mi}].parts[${pi}]`,
        );
        return;
      }
      blocks.push({ image: { format, source: { bytes: part.data } } });
    } else if (isToolCall(part)) {
      blocks.push({ toolUse: { toolUseId: part.id, name: part.name, input: part.args } });
    } else if (isToolResult(part)) {
      blocks.push({
        toolResult: {
          toolUseId: part.id,
          status: part.isError === true ? "error" : "success",
          content: bedrockToolResultBlocks(part, msg),
        },
      });
    }
  });
  return blocks;
}

/** Revive Bedrock toolResult.content: original array if stashed, else wrap the string. */
function bedrockToolResultBlocks(part: ToolResult, msg: Message): BedrockToolResultContent[] {
  const stashed = msg.extra?.[`bedrock.toolResult.${part.id}`];
  if (Array.isArray(stashed)) return stashed as BedrockToolResultContent[];
  try {
    const parsed: unknown = JSON.parse(part.content);
    if (typeof parsed === "object" && parsed !== null) return [{ json: parsed }];
  } catch {
    // not JSON; fall through to text
  }
  return [{ text: part.content }];
}

function asDocumentPayload(doc: unknown): unknown {
  // A stashed raw Bedrock block is { document: {...} }; unwrap to the inner value.
  if (typeof doc === "object" && doc !== null && "document" in doc) {
    return (doc as { document: unknown }).document;
  }
  return doc;
}

// ── shared tail ──────────────────────────────────────────────────────────────

function result<T>(output: T, priors: Priors, notes: NoteCollector): ConvertResult<T> {
  return { output, notes: [...priors, ...notes.notes] };
}

function readExtraString(msg: Message, key: string): string | undefined {
  const value = msg.extra?.[key];
  return typeof value === "string" ? value : undefined;
}

/** Serialize tool-call args back to the JSON string OpenAI expects. */
function stringifyArgs(args: unknown): string {
  if (typeof args === "string") return args;
  return JSON.stringify(args ?? {});
}

// Keep these part types referenced for downstream importers/type clarity.
export type { TextPart, ToolCall };
