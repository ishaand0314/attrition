/**
 * convert(from, to, payload) — the one public composition.
 *
 * Pipeline: parse -> validate -> serialize. Notes from all three stages are
 * concatenated in order, so the returned `ConvertResult.notes` is the complete
 * story of what happened to the conversation.
 *
 * Adding a provider is one entry in PARSERS and one in SERIALIZERS — no pairwise
 * converters, no N^2.
 */

import type { Conversation } from "./message.js";
import type { ConvertResult, LossNote } from "./notes.js";
import {
  type ParseResult,
  parseAnthropic,
  parseBedrock,
  parseGemini,
  parseOpenAI,
} from "./parse.js";
import { type SerializeOptions, toAnthropic, toBedrock, toGemini, toOpenAI } from "./serialize.js";
import { validate } from "./validate.js";

export const PROVIDERS = ["openai", "anthropic", "gemini", "bedrock"] as const;
export type Provider = (typeof PROVIDERS)[number];

export function isProvider(value: string): value is Provider {
  return (PROVIDERS as readonly string[]).includes(value);
}

const PARSERS: Record<Provider, (payload: unknown) => ParseResult> = {
  openai: parseOpenAI,
  anthropic: parseAnthropic,
  gemini: parseGemini,
  bedrock: parseBedrock,
};

type Serializer = (
  conversation: Conversation,
  priors: readonly LossNote[],
  options: SerializeOptions,
) => ConvertResult<unknown>;

const SERIALIZERS: Record<Provider, Serializer> = {
  openai: toOpenAI,
  anthropic: toAnthropic,
  gemini: toGemini,
  bedrock: toBedrock,
};

export type ConvertOptions = SerializeOptions;

/**
 * Convert a conversation `payload` from one provider format to another.
 * Never throws on a lossy conversion — the losses are in `.notes`.
 *
 * `options.textOnly` declares the target *model* can't take images (many can't),
 * which turns image parts into a `loss` — the "image into a text-only target"
 * seam. It's a property of the destination model, not the format, so you pass it.
 */
export function convert(
  from: Provider,
  to: Provider,
  payload: unknown,
  options: ConvertOptions = {},
): ConvertResult<unknown> {
  const { conversation, collector } = PARSERS[from](payload);
  validate(conversation, collector);
  return SERIALIZERS[to](conversation, collector.notes, options);
}
