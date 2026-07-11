/**
 * attrition — convert a conversation between OpenAI, Anthropic, Gemini, and AWS
 * Bedrock message formats, and get a report of exactly what did not survive the
 * trip.
 *
 * The library is the product; the CLI (cli.ts) is a thin shell over it.
 *
 *   import { convert } from "llm-attrition";
 *   const { output, notes } = convert("openai", "anthropic", payload);
 */

// The canonical IR (hand-synced with labkit — see message.ts).
export type {
  Conversation,
  ImagePart,
  Message,
  Part,
  Role,
  TextPart,
  ToolCall,
  ToolResult,
} from "./message.js";
export { isImage, isText, isToolCall, isToolResult } from "./message.js";

// The lossiness report.
export type { ConvertResult, LossNote, Severity } from "./notes.js";
export { hasLoss, NoteCollector } from "./notes.js";

// Parsers, validator, serializers, and the one public composition.
export {
  type ParseResult,
  parseAnthropic,
  parseBedrock,
  parseGemini,
  parseOpenAI,
} from "./parse.js";
export { validate } from "./validate.js";
export { type SerializeOptions, toAnthropic, toBedrock, toGemini, toOpenAI } from "./serialize.js";
export { type ConvertOptions, convert, isProvider, PROVIDERS, type Provider } from "./convert.js";
