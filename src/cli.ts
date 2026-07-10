#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { argv } from "node:process";
import { fileURLToPath } from "node:url";
import type { Command, CommandContext } from "./cli-router.js";
import { run } from "./cli-router.js";
import { PROVIDERS, type Provider, convert, isProvider } from "./convert.js";
import { type LossNote, hasLoss } from "./notes.js";

/**
 * CLI entry. Uses the shared-style router (cli-router.ts) for a consistent UX
 * with the rest of the 7-in-7 tools (--json, --help, consistent errors).
 *
 * Usage:
 *   attrition convert --from openai --to anthropic --file conv.json
 *   attrition convert --from openai --to gemini   --file conv.json --json
 *   attrition convert --from openai --to anthropic --file conv.json --strict
 *   cat conv.json | attrition convert --from openai --to anthropic
 *
 * Default output: the converted payload as pretty JSON on stdout; the notes on
 * stderr (so a redirect captures a clean payload). A zero-note conversion prints
 * nothing to stderr. --json emits { output, notes } together on stdout.
 * --strict exits 1 if any note is a "loss".
 */

/** Thrown for bad user input; caught to print a one-line error. */
class UsageError extends Error {}

/** Injected I/O so the command is testable without touching the real process. */
export interface CliIo {
  readFile(path: string): string;
  readStdin(): string;
  stdinIsTty: boolean;
  writeOut(text: string): void;
  writeErr(text: string): void;
}

const providerList = PROVIDERS.join(", ");

function requireProvider(value: string | boolean | undefined, flag: string): Provider {
  if (typeof value !== "string") {
    throw new UsageError(`--${flag} is required, one of: ${providerList}`);
  }
  if (!isProvider(value)) {
    throw new UsageError(`--${flag} must be one of: ${providerList} (got "${value}")`);
  }
  return value;
}

function readPayloadText(flags: Record<string, string | boolean>, io: CliIo): string {
  const file = flags.file;
  if (file !== undefined) {
    if (typeof file !== "string") {
      throw new UsageError("--file requires a filename, e.g. --file conv.json");
    }
    try {
      return io.readFile(file);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new UsageError(`Cannot read --file "${file}": ${msg}`);
    }
  }
  if (!io.stdinIsTty) {
    return io.readStdin();
  }
  throw new UsageError("Provide input: --file conv.json, or pipe a conversation on stdin");
}

function formatNote(note: LossNote): string {
  const badge = note.severity.toUpperCase().padEnd(7);
  const where = note.path ? `  (${note.path})` : "";
  return `${badge} ${note.message}${where}`;
}

/**
 * The `convert` command body, exposed for testing. Returns the intended exit
 * code (0 ok, 1 on usage error or on a "loss" under --strict).
 */
export function runConvertCommand(ctx: CommandContext, io: CliIo): number {
  const { flags } = ctx;
  let from: Provider;
  let to: Provider;
  let text: string;
  try {
    from = requireProvider(flags.from, "from");
    to = requireProvider(flags.to, "to");
    text = readPayloadText(flags, io);
  } catch (err) {
    if (err instanceof UsageError) {
      io.writeErr(`${err.message}\n`);
      return 1;
    }
    throw err;
  }

  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    io.writeErr(`Input is not valid JSON: ${msg}\n`);
    return 1;
  }

  const { output, notes } = convert(from, to, payload, { textOnly: flags["text-only"] === true });

  if (flags.json) {
    io.writeOut(`${JSON.stringify({ output, notes }, null, 2)}\n`);
  } else {
    io.writeOut(`${JSON.stringify(output, null, 2)}\n`);
    // Notes go to stderr; silence means clean (no "no issues found" line).
    if (notes.length > 0) {
      io.writeErr(`\n${notes.map(formatNote).join("\n")}\n`);
    }
  }

  if (flags.strict && hasLoss(notes)) {
    io.writeErr("\n--strict: conversion has losses (exit 1)\n");
    return 1;
  }
  return 0;
}

const realIo: CliIo = {
  readFile: (path) => readFileSync(path, "utf8"),
  readStdin: () => readFileSync(0, "utf8"),
  stdinIsTty: process.stdin.isTTY === true,
  writeOut: (text) => process.stdout.write(text),
  writeErr: (text) => process.stderr.write(text),
};

const convertCommand: Command = {
  name: "convert",
  summary: "Convert a conversation between provider formats and report what changed",
  run(ctx) {
    const code = runConvertCommand(ctx, realIo);
    if (code !== 0) process.exitCode = code;
  },
};

/** Only run the CLI when invoked directly, not when imported by tests. */
function isMain(): boolean {
  const entry = argv[1];
  return entry !== undefined && entry === fileURLToPath(import.meta.url);
}

if (isMain()) {
  await run(
    {
      name: "attrition",
      description: `Cross-lab chat message format translator (${providerList})`,
      commands: [convertCommand],
      booleanFlags: ["json", "strict", "text-only"],
    },
    argv.slice(2),
  );
}
