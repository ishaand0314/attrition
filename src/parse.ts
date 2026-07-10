/**
 * Parsers: a provider's wire payload -> the canonical `Conversation` IR.
 *
 * One parser per provider. Adding a lab means adding one function here (and one
 * in serialize.ts) — never a pairwise converter.
 *
 * Parsers are permissive readers: they lift whatever they recognize into the
 * IR and record `info` notes for lossless-but-notable normalizations (role
 * renames, synthesized ids, hoisted system prompts, stashed provider-only
 * fields). They do NOT enforce target-format invariants — that is validate.ts's
 * job, run once between parse and serialize.
 */

import type {
  Conversation,
  Message,
  Part,
  Role,
  TextPart,
  ToolCall,
  ToolResult,
} from "./message.js";
import { NoteCollector } from "./notes.js";

/** A parsed conversation plus the notes the parser emitted. */
export interface ParseResult {
  conversation: Conversation;
  collector: NoteCollector;
}

// ── small shared helpers ─────────────────────────────────────────────────────

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/** Parse a data: URL into { mimeType, data(base64) }, or null if not one. */
function parseDataUrl(url: string): { mimeType: string; data: string } | null {
  const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
  if (!match) return null;
  const mimeType = match[1];
  const data = match[2];
  if (mimeType === undefined || data === undefined) return null;
  return { mimeType, data };
}

// ── OpenAI ───────────────────────────────────────────────────────────────────
//
// Shape: { messages: [{ role, content, name?, tool_calls?, tool_call_id? }] }
// - system lives IN the messages array (role: "system")
// - content is a string or an array of { type: "text" | "image_url", ... }
// - assistant tool calls: message.tool_calls[] = { id, function: { name, arguments } }
// - tool results: role "tool", with tool_call_id and string content

export function parseOpenAI(payload: unknown): ParseResult {
  const notes = new NoteCollector();
  const root = asRecord(payload);
  const rawMessages = asArray(root.messages);

  let system: string | undefined;
  const systemChunks: string[] = [];
  const messages: Message[] = [];

  rawMessages.forEach((rawMsg, i) => {
    const msg = asRecord(rawMsg);
    const role = asString(msg.role);

    if (role === "system") {
      // Hoist system out of the array. Flag if it isn't the first message.
      if (messages.length > 0) {
        notes.warning(
          "Hoisted a system message that appeared after other messages into the top-level system prompt",
          `messages[${i}]`,
        );
      }
      systemChunks.push(contentToText(msg.content));
      return;
    }

    if (role === "tool") {
      // OpenAI tool results are their own messages, keyed by tool_call_id.
      const toolResult: ToolResult = {
        type: "tool_result",
        id: asString(msg.tool_call_id),
        content: contentToText(msg.content),
      };
      messages.push({ role: "tool", parts: [toolResult] });
      return;
    }

    const parts: Part[] = openaiContentParts(msg.content);

    // assistant tool_calls
    for (const rawCall of asArray(msg.tool_calls)) {
      const call = asRecord(rawCall);
      const fn = asRecord(call.function);
      parts.push({
        type: "tool_call",
        id: asString(call.id),
        name: asString(fn.name),
        args: parseJsonArgs(fn.arguments),
      });
    }

    const irRole: Role = role === "assistant" ? "assistant" : "user";
    const message: Message = { role: irRole, parts };

    // OpenAI `name` has no IR home — stash it, note it.
    if (typeof msg.name === "string") {
      message.extra = { "openai.name": msg.name };
      notes.info(`Preserved OpenAI \`name\` field ("${msg.name}") in extra`, `messages[${i}]`);
    }

    messages.push(message);
  });

  if (systemChunks.length > 1) {
    notes.warning(`Merged ${systemChunks.length} system messages into one`);
  }
  if (systemChunks.length > 0) {
    system = systemChunks.join("\n\n");
  }

  return { conversation: buildConversation(system, messages), collector: notes };
}

function openaiContentParts(content: unknown): Part[] {
  if (typeof content === "string") {
    return content.length > 0 ? [{ type: "text", text: content }] : [];
  }
  const parts: Part[] = [];
  for (const rawChunk of asArray(content)) {
    const chunk = asRecord(rawChunk);
    if (chunk.type === "text") {
      parts.push({ type: "text", text: asString(chunk.text) });
    } else if (chunk.type === "image_url") {
      const url = asString(asRecord(chunk.image_url).url);
      const parsed = parseDataUrl(url);
      if (parsed) {
        parts.push({ type: "image", mimeType: parsed.mimeType, data: parsed.data });
      } else {
        // A remote image URL — keep the URL as the base64 field is meaningless;
        // treat mimeType as unknown. Round-trips as an image_url either way.
        parts.push({ type: "image", mimeType: "image/*", data: url });
      }
    }
  }
  return parts;
}

// ── Anthropic ─────────────────────────────────────────────────────────────────
//
// Shape: { system?: string, messages: [{ role, content }] }
// - system is a top-level field (string)
// - content is a string or an array of blocks:
//     { type: "text", text }
//     { type: "image", source: { type: "base64", media_type, data } }
//     { type: "tool_use", id, name, input }
//     { type: "tool_result", tool_use_id, content, is_error? }

export function parseAnthropic(payload: unknown): ParseResult {
  const notes = new NoteCollector();
  const root = asRecord(payload);

  const system =
    typeof root.system === "string" && root.system.length > 0 ? root.system : undefined;
  const messages: Message[] = [];

  for (const rawMsg of asArray(root.messages)) {
    const msg = asRecord(rawMsg);
    const role = asString(msg.role);
    const parts: Part[] = [];
    const toolResults: Message[] = [];

    for (const block of anthropicBlocks(msg.content)) {
      const b = asRecord(block);
      switch (b.type) {
        case "text":
          parts.push({ type: "text", text: asString(b.text) });
          break;
        case "image": {
          const source = asRecord(b.source);
          parts.push({
            type: "image",
            mimeType: asString(source.media_type),
            data: asString(source.data),
          });
          break;
        }
        case "tool_use":
          parts.push({
            type: "tool_call",
            id: asString(b.id),
            name: asString(b.name),
            args: b.input,
          });
          break;
        case "tool_result": {
          // Anthropic nests tool results inside a user message; the IR gives
          // them their own `role: "tool"` message so serializers can relocate.
          const result: ToolResult = {
            type: "tool_result",
            id: asString(b.tool_use_id),
            content: anthropicToolResultContent(b.content),
          };
          if (b.is_error === true) result.isError = true;
          toolResults.push({ role: "tool", parts: [result] });
          break;
        }
        default:
          break;
      }
    }

    const irRole: Role = role === "assistant" ? "assistant" : "user";
    if (parts.length > 0) messages.push({ role: irRole, parts });
    // tool_result blocks become their own tool messages, after the text/tool_use.
    messages.push(...toolResults);
  }

  return { conversation: buildConversation(system, messages), collector: notes };
}

function anthropicBlocks(content: unknown): unknown[] {
  if (typeof content === "string") {
    return content.length > 0 ? [{ type: "text", text: content }] : [];
  }
  return asArray(content);
}

function anthropicToolResultContent(content: unknown): string {
  if (typeof content === "string") return content;
  // Anthropic allows tool_result content to be an array of blocks; flatten text.
  const texts: string[] = [];
  for (const rawBlock of asArray(content)) {
    const block = asRecord(rawBlock);
    if (block.type === "text") texts.push(asString(block.text));
  }
  return texts.join("\n");
}

// ── Gemini ────────────────────────────────────────────────────────────────────
//
// Shape: { systemInstruction?: { parts: [{ text }] }, contents: [{ role, parts }] }
// - role is "user" or "model" (never "assistant"); tool turns use "user"/"function"
// - parts: { text } | { inlineData: { mimeType, data } }
//          | { functionCall: { name, args } }
//          | { functionResponse: { name, response } }
// - NO tool-call ids exist — calls and responses are matched by function name.

export function parseGemini(payload: unknown): ParseResult {
  const notes = new NoteCollector();
  const root = asRecord(payload);

  const system = geminiSystemText(root.systemInstruction);
  const messages: Message[] = [];
  let renamedModel = false;
  let synthesizedId = false;

  // Gemini matches functionResponse -> functionCall by name and order. We
  // synthesize ids so the IR (which is id-based) can pair them: each new call
  // name gets an id; the next response with that name reuses it.
  const pendingIdByName = new Map<string, string[]>();
  let synthCounter = 0;

  asArray(root.contents).forEach((rawContent, ci) => {
    const content = asRecord(rawContent);
    const rawRole = asString(content.role);
    const parts: Part[] = [];

    for (const rawPart of asArray(content.parts)) {
      const part = asRecord(rawPart);
      if (typeof part.text === "string") {
        parts.push({ type: "text", text: part.text });
      } else if (part.inlineData !== undefined) {
        const inline = asRecord(part.inlineData);
        parts.push({
          type: "image",
          mimeType: asString(inline.mimeType),
          data: asString(inline.data),
        });
      } else if (part.functionCall !== undefined) {
        const call = asRecord(part.functionCall);
        const name = asString(call.name);
        const id = `gemini-call-${synthCounter++}`;
        synthesizedId = true;
        const queue = pendingIdByName.get(name) ?? [];
        queue.push(id);
        pendingIdByName.set(name, queue);
        const toolCall: ToolCall = { type: "tool_call", id, name, args: call.args };
        parts.push(toolCall);
      } else if (part.functionResponse !== undefined) {
        const response = asRecord(part.functionResponse);
        const name = asString(response.name);
        const queue = pendingIdByName.get(name) ?? [];
        // Gemini pairs responses to calls only by function name; when more than
        // one call of this name is open, the call->result pairing is ambiguous
        // and this by-order guess may be wrong. Flag it rather than pair silently.
        if (queue.length > 1) {
          notes.warning(
            `Ambiguous Gemini tool-result pairing: ${queue.length} open calls named "${name}"; paired by order, which may be wrong`,
            `contents[${ci}]`,
          );
        }
        const id = queue.shift() ?? `gemini-call-${synthCounter++}`;
        pendingIdByName.set(name, queue);
        const result: ToolResult = {
          type: "tool_result",
          id,
          name,
          content: geminiResponseContent(response.response),
        };
        const toolMessage: Message = { role: "tool", parts: [result] };
        // Gemini responses are structured objects but the IR content is a string.
        // Stash the original object so toGemini can revive it byte-for-byte
        // (JSON.stringify∘JSON.parse is NOT identity: "1.0"->1, key reordering).
        if (typeof response.response === "object" && response.response !== null) {
          toolMessage.extra = { [`gemini.functionResponse.${id}`]: response.response };
        }
        messages.push(toolMessage);
      }
    }

    if (parts.length === 0) return;

    let irRole: Role;
    if (rawRole === "model") {
      irRole = "assistant";
      renamedModel = true;
    } else {
      irRole = "user";
    }
    messages.push({ role: irRole, parts });
  });

  if (renamedModel) {
    notes.info('Renamed Gemini role "model" to "assistant"');
  }
  if (synthesizedId) {
    notes.info(
      "Synthesized tool-call ids for Gemini (matched calls to responses by name and order)",
    );
  }

  return { conversation: buildConversation(system, messages), collector: notes };
}

function geminiSystemText(systemInstruction: unknown): string | undefined {
  if (systemInstruction === undefined) return undefined;
  const instr = asRecord(systemInstruction);
  const texts: string[] = [];
  for (const rawPart of asArray(instr.parts)) {
    const part = asRecord(rawPart);
    if (typeof part.text === "string") texts.push(part.text);
  }
  const joined = texts.join("\n\n");
  return joined.length > 0 ? joined : undefined;
}

function geminiResponseContent(response: unknown): string {
  if (typeof response === "string") return response;
  // Gemini responses are objects; stringify deterministically for the IR.
  return JSON.stringify(response ?? {});
}

// ── Bedrock (AWS Bedrock Converse API) ────────────────────────────────────────
//
// Shape: { system?: [{ text }], messages: [{ role, content }] }
// - roles are only "user" | "assistant"; there is no system or tool role
// - system is a top-level ARRAY of blocks (not a string)
// - content blocks: { text } | { image: { format, source: { bytes } } }
//                 | { document: { name, format, source: { bytes } } }
//                 | { toolUse: { toolUseId, name, input } }
//                 | { toolResult: { toolUseId, status?, content: [...] } }
// - image `format` is a BARE ENUM ("png"), not a MIME type
// - toolResult.content is an ARRAY that can hold { json: <object> } (structured)
// - document blocks have no IR home; they are dropped with a `loss` (see below)

const BEDROCK_IMAGE_FORMAT_TO_MIME: Record<string, string> = {
  png: "image/png",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
};

export function parseBedrock(payload: unknown): ParseResult {
  const notes = new NoteCollector();
  const root = asRecord(payload);

  // system: an array of blocks; join the text ones, drop non-text with a warning.
  const systemBlocks = asArray(root.system);
  const systemTexts: string[] = [];
  for (const rawBlock of systemBlocks) {
    const block = asRecord(rawBlock);
    if (typeof block.text === "string") {
      systemTexts.push(block.text);
    } else {
      notes.warning("Dropped a non-text Bedrock system block");
    }
  }
  if (systemTexts.length > 1) {
    notes.info(`Joined ${systemTexts.length} Bedrock system blocks into one`);
  }
  const system = systemTexts.length > 0 ? systemTexts.join("\n\n") : undefined;

  const messages: Message[] = [];
  let mappedImageFormat = false;

  asArray(root.messages).forEach((rawMsg, mi) => {
    const msg = asRecord(rawMsg);
    const rawRole = asString(msg.role);
    const parts: Part[] = [];
    const toolResults: Message[] = [];
    const documents: unknown[] = [];

    for (const rawBlock of asArray(msg.content)) {
      const block = asRecord(rawBlock);
      if (typeof block.text === "string") {
        parts.push({ type: "text", text: block.text });
      } else if (block.image !== undefined) {
        const image = asRecord(block.image);
        const format = asString(image.format);
        const source = asRecord(image.source);
        const mime = BEDROCK_IMAGE_FORMAT_TO_MIME[format];
        if (mime !== undefined) mappedImageFormat = true;
        if (typeof source.bytes === "string") {
          parts.push({ type: "image", mimeType: mime ?? "image/*", data: source.bytes });
        } else {
          // s3Location (or anything non-inline): can't be embedded as base64.
          notes.loss(
            "Bedrock image is not inline bytes (e.g. an S3 reference); cannot inline",
            `messages[${mi}]`,
          );
        }
      } else if (block.document !== undefined) {
        // The IR has no document part. Drop it (loss) but stash for round-trip.
        const doc = asRecord(block.document);
        notes.loss(
          `Dropped a Bedrock document block (name: "${asString(doc.name)}", format: "${asString(doc.format)}"); the IR has no document type`,
          `messages[${mi}]`,
        );
        documents.push(rawBlock);
      } else if (block.toolUse !== undefined) {
        const toolUse = asRecord(block.toolUse);
        parts.push({
          type: "tool_call",
          id: asString(toolUse.toolUseId),
          name: asString(toolUse.name),
          args: toolUse.input,
        });
      } else if (block.toolResult !== undefined) {
        const tr = asRecord(block.toolResult);
        const { content, hadJson } = bedrockToolResultContent(tr.content, notes, mi);
        const result: ToolResult = {
          type: "tool_result",
          id: asString(tr.toolUseId),
          content,
        };
        if (tr.status === "error") result.isError = true;
        const message: Message = { role: "tool", parts: [result] };
        if (hadJson) {
          // Stash the original content array so toBedrock can revive it losslessly.
          notes.info(
            "Serialized structured (JSON) Bedrock tool-result content to a string; original preserved for round-trip",
            `messages[${mi}]`,
          );
          message.extra = { [`bedrock.toolResult.${result.id}`]: tr.content };
        }
        toolResults.push(message);
      }
    }

    let irRole: Role;
    if (rawRole === "assistant") {
      irRole = "assistant";
    } else if (rawRole === "user") {
      irRole = "user";
    } else {
      irRole = "user";
      notes.warning(`Treated unknown Bedrock role "${rawRole}" as "user"`, `messages[${mi}]`);
    }

    if (parts.length > 0) {
      const message: Message = { role: irRole, parts };
      if (documents.length > 0) message.extra = { "bedrock.documents": documents };
      messages.push(message);
    }
    // A document-only message has no IR part to carry the stash, so its documents
    // are truly lost (already reported as `loss` above). Not silently "fixed".
    messages.push(...toolResults);
  });

  if (mappedImageFormat) {
    notes.info("Mapped Bedrock image format enum to MIME type (e.g. png -> image/png)");
  }

  return { conversation: buildConversation(system, messages), collector: notes };
}

/** Flatten a Bedrock toolResult.content[] array to a string; note structured JSON. */
function bedrockToolResultContent(
  content: unknown,
  notes: NoteCollector,
  mi: number,
): { content: string; hadJson: boolean } {
  if (typeof content === "string") return { content, hadJson: false };
  const texts: string[] = [];
  let hadJson = false;
  for (const rawEntry of asArray(content)) {
    const entry = asRecord(rawEntry);
    if (typeof entry.text === "string") {
      texts.push(entry.text);
    } else if (entry.json !== undefined) {
      texts.push(JSON.stringify(entry.json));
      hadJson = true;
    } else if (entry.image !== undefined || entry.document !== undefined) {
      notes.loss("Dropped non-text content in a Bedrock tool result", `messages[${mi}]`);
    }
  }
  return { content: texts.join("\n"), hadJson };
}

// ── shared tail ──────────────────────────────────────────────────────────────

function buildConversation(system: string | undefined, messages: Message[]): Conversation {
  return system === undefined ? { messages } : { system, messages };
}

/** Coerce a provider `content` (string or block array) to plain text. */
function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  const texts: string[] = [];
  for (const rawChunk of asArray(content)) {
    const chunk = asRecord(rawChunk);
    if (typeof chunk.text === "string") texts.push(chunk.text);
  }
  return texts.join("\n");
}

/** Parse an OpenAI tool-call `arguments` string (JSON) into a value. */
function parseJsonArgs(args: unknown): unknown {
  if (typeof args !== "string") return args;
  try {
    return JSON.parse(args);
  } catch {
    return args;
  }
}

// Re-export so callers importing from parse can name the text part type.
export type { TextPart };
