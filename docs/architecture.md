# Architecture

## One IR, not N² converters

Three providers means six directed conversions today; a fifth provider would mean
twenty. Writing a converter per pair does not scale. So nothing here converts one
format directly to another.

Every provider **parses** into one canonical `Conversation`, and every provider
**serializes** out of it:

```
OpenAI     ──parse──┐                    ┌──serialize──▶ OpenAI
Anthropic  ──parse──┼──▶  Conversation  ─┼──serialize──▶ Anthropic
Gemini     ──parse──┘     (canonical IR) └──serialize──▶ Gemini
```

Adding a provider is exactly two functions: one `parse` and one `serialize`.
Never N more.

## The pipeline: parse → validate → serialize

`convert(from, to, payload)` composes three stages:

1. **parse** (`src/parse.ts`) — the source provider's wire payload becomes a
   `Conversation`. Parsers are permissive readers. They record `info` notes for
   lossless-but-notable normalizations (a role rename, a synthesized id, a
   hoisted system prompt, a stashed provider-only field). They do not enforce any
   target's rules.

2. **validate** (`src/validate.ts`) — a provider-agnostic step that fixes
   IR-level invariant violations **once**, before any serializer runs. Today that
   is a single rule: a tool result must answer a tool call that exists. This runs
   here, not in each serializer, so all three serializers agree on the behavior
   instead of each inventing its own.

3. **serialize** (`src/serialize.ts`) — the `Conversation` becomes the target
   provider's wire payload. Serializers assume a valid IR and handle
   format-specific reshaping (merging consecutive same-role messages, dropping
   empty blocks, dropping images for a text-only target), recording a note for
   each.

Notes from all three stages are concatenated in order, so the returned
`ConvertResult.notes` is the complete story of the conversion.

## The IR

`src/message.ts` holds the canonical types. Two modeling calls matter:

- **The system prompt is hoisted out of the message array.** OpenAI carries it as
  an in-array `role: "system"` message; Anthropic and Gemini take it as a
  top-level field. The IR holds the *meaning* in `Conversation.system`, and each
  serializer decides where it physically goes.

- **Tool results get their own `role: "tool"` message.** OpenAI already models
  them that way; Anthropic nests them inside a user turn and Gemini inside a
  user/function turn. Normalizing to a distinct role lets each serializer relocate
  them without special-casing where they came from.

## The lossiness report is the product

`src/notes.ts` defines `LossNote` (`severity`, `message`, `path`) and
`ConvertResult<T>` (`output`, `notes`). Every conversion returns both. The rule
is: never throw on a lossy conversion, never convert silently.

- `info` — faithful and reversible.
- `warning` — lossy but necessary reshaping.
- `loss` — content did not survive.

`--strict` in the CLI exits non-zero if any note is a `loss`, which is what makes
this runnable as a CI gate.

## One decision worth calling out: never fabricate structure

When the validator finds a tool result with no matching call, it **drops** it and
records a `loss`. It does **not** synthesize a fake preceding tool call to make
the conversation pass a schema. Fabricating a call the model never made is an
invisible corruption: the downstream model then conditions on a call that didn't
happen. That is strictly worse than dropping content, and it is exactly the kind
of quiet untrustworthiness this tool exists to prevent. When a conversion can't be
both faithful and valid, it prefers valid and honest over faithful and invented.

## Tool choices

- **pnpm** — fast, disk-efficient installs with a strict lockfile.
- **Strict TypeScript** — `strict`, `noUncheckedIndexedAccess`,
  `verbatimModuleSyntax`. The IR is small and precise; there are no `any`s and no
  non-null assertions in `src/`.
- **Biome** — one fast binary for lint and format.
- **Vitest** — aliased to source (tests import from `src` directly), so the suite
  runs with no build and always reflects the latest code.

## Conventions

- The package exposes a **library** (`src/index.ts`) and a **CLI** (`src/cli.ts`).
  The library is the product; the CLI is a thin shell over it.
- The CLI router (`src/cli-router.ts`) gives the CLI a small, consistent
  surface — `--json` / `--help` and uniform error handling — kept separate from
  the conversion logic it drives.
- Pure functions, no hidden global state, so everything is trivially testable.
- The tests double as the spec: the seams table in the README is the contract,
  and every row has a test.
