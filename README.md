# attrition

Convert a conversation between OpenAI, Anthropic, Gemini, and AWS Bedrock message
formats, and get a report of exactly what did not survive the trip.

*attrition* names what the tool measures: the content that falls away when a
conversation crosses from one provider's format into another's.

Gateways do this conversion internally and lock it behind their client. This
ships it as a standalone you drop into an existing codebase, and it tells you
what it dropped.

## Why you'd use it

You have a conversation in one provider's format and need it in another: you're
switching providers, running the same history through two models, or building a
gateway of your own. The conversion is never perfectly clean. Roles get renamed,
system prompts move, tool-call ids get remapped, images get dropped by text-only
models. Most tools do this silently. This one hands you a list of every change.

```ts
import { convert } from "attrition";

const { output, notes } = convert("openai", "anthropic", conversation);
// output: the Anthropic-shaped payload
// notes:  every info / warning / loss, in order, with a path
```

The rule: **never throw on a lossy conversion, never convert silently.**

## Install

Not yet published to npm. Clone and build from source:

```bash
git clone https://github.com/ishaand0314/attrition.git
cd attrition
pnpm install && pnpm build
node dist/cli.js convert --from openai --to anthropic --file conv.json
```

The examples below use `attrition` as the command name; that is the CLI's own
name (`node dist/cli.js` while running from source).

## Use it

```bash
# convert a file, payload to stdout, notes to stderr
attrition convert --from openai --to anthropic --file conv.json

# pipe it in
cat conv.json | attrition convert --from openai --to gemini

# machine-readable: { output, notes } together on stdout
attrition convert --from openai --to anthropic --file conv.json --json

# CI gate: exit 1 if any note is a "loss"
attrition convert --from openai --to anthropic --file conv.json --strict

# the target model can't take images: drop them and report the loss
attrition convert --from openai --to anthropic --file conv.json --text-only
```

By default the converted payload goes to **stdout** (so `> out.json` captures a
clean payload) and the notes go to **stderr**. A conversion with no notes prints
nothing to stderr: silence means clean.

## Playground

A single static page to paste a conversation, pick from/to providers, and watch
the lossiness report update live:

```bash
pnpm playground   # builds, then serves at http://localhost:5050/playground/
```

It imports the real compiled `dist/index.js`, so the notes it shows are the
actual conversion output, not a copy that could drift. Three preloaded examples
each surface a different severity (a system merge, a tool-call id drop, an image
into a text-only model).

## The lossiness report

Every conversion returns `{ output, notes }`. A note has a severity:

- **info**: a faithful, reversible change (e.g. renaming `assistant` to `model`).
- **warning**: a lossy-but-necessary reshaping (e.g. merging system messages).
- **loss**: content did not survive (e.g. an image into a text-only target).

`--strict` exits non-zero if any note is a `loss`, so a broken conversion fails
your CI instead of silently shipping a mangled history.

## The seams

Every place the three formats disagree, and what this tool does about it. This
table is the spec: each row has a test.

| Seam | Behavior | Severity |
| --- | --- | --- |
| OpenAI keeps system in the message array; Anthropic/Gemini take it top-level | Hoisted out of the array; each serializer places it | info / warning |
| A system message that appears after other messages (OpenAI allows it) | Hoisted and merged; order change flagged | warning |
| Multiple system messages into Anthropic/Gemini | Merged into one | warning |
| `assistant` role into Gemini | Renamed to `model` (fully reversible) | info |
| OpenAI/Anthropic tool-call ids into Gemini | Gemini has no ids; matched by name, id dropped | info |
| Gemini `functionCall`/`functionResponse` into OpenAI/Anthropic | Ids synthesized, calls paired to results by name and order | info |
| Tool result with no matching call | Dropped (never fabricate a fake call) | loss |
| A dropped orphan that empties its message (Anthropic rejects empty) | The empty message is dropped too | warning |
| Tool call with no result (normal mid-loop state) | Kept; legal everywhere | none |
| `tool` role placement (own message vs nested in a user turn) | Relocated into a user turn for Anthropic/Gemini | info |
| Consecutive same-role messages (Anthropic rejects) | Merged | warning |
| Empty content block (Anthropic rejects) | Dropped | warning |
| First message is not `user` (Anthropic/Bedrock require it) | Flagged; serializer can't invent a turn | warning |
| System-only conversation into Anthropic/Bedrock (empty messages array) | Flagged | warning |
| Image part into a text-only target model (`--text-only`) | Dropped | loss |
| Remote image URL into a base64-only target (Anthropic/Gemini/Bedrock) | No bytes to embed; dropped | loss |
| Image whose MIME has no Bedrock enum (only png/jpeg/gif/webp) | Dropped | loss |
| OpenAI `name` field (no target equivalent) | Kept in `extra` | info |
| Bedrock `document` blocks (no IR equivalent) | Dropped; kept in `extra` for round-trip | loss |

## Architecture: one IR, not twenty converters

Adding a provider is one parser and one serializer. It is never a new pairwise
converter (four providers would be twelve of those; Bedrock was added as exactly
one parser and one serializer, no change to any other provider). Every provider
parses into one canonical `Conversation`, and every provider serializes out of it.

```
OpenAI     ──parse──┐                    ┌──serialize──▶ OpenAI
Anthropic  ──parse──┤                    ├──serialize──▶ Anthropic
Gemini     ──parse──┼──▶  Conversation  ─┼──serialize──▶ Gemini
Bedrock    ──parse──┘     (canonical IR) └──serialize──▶ Bedrock
```

Between parse and serialize, one provider-agnostic **validate** step fixes IR
invariant violations (an orphan tool result) once, so all three serializers can
assume a well-formed conversation. See [docs/architecture.md](docs/architecture.md).

## In your own code

```ts
import { convert, hasLoss } from "attrition";

const { output, notes } = convert("gemini", "openai", conversation);

if (hasLoss(notes)) {
  for (const n of notes.filter((x) => x.severity === "loss")) {
    console.warn(`lost: ${n.message} (${n.path ?? "?"})`);
  }
}

send(output);
```

The canonical IR (`Conversation`, `Message`, `Part`, ...) is exported too, if you
want to build against it directly.

> **Note on the IR.** The canonical `Message` type in `src/message.ts` is the
> same shape used by labkit (this series' day-1 tool). There is no shared npm
> package between the two repos; the type is kept in sync by hand. If you change
> it here, mirror it in labkit.

## Scope

Day 2 of a 7-in-7 build. It converts **conversation content** (messages, roles,
text, images, tool calls and results, the system prompt). It does not touch
request config that isn't part of the conversation: sampling params,
`tool_choice`/`response_format`, `cache_control`, `safetySettings`, streaming, or
audio parts. Those are config, not conversation.
