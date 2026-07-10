import { describe, expect, it } from "vitest";
import { type CliIo, runConvertCommand } from "../src/cli.js";

/**
 * The CLI's convert command, driven with a fake IO so we can assert exit codes
 * and the stdout/stderr split without spawning a process or building.
 */

interface Captured {
  out: string;
  err: string;
  io: CliIo;
}

/** A fake IO whose stdin is a piped conversation payload. */
function fakeIo(stdin: string, files: Record<string, string> = {}): Captured {
  const cap: Captured = {
    out: "",
    err: "",
    io: {
      readFile(path) {
        const file = files[path];
        if (file === undefined) throw new Error("ENOENT");
        return file;
      },
      readStdin: () => stdin,
      stdinIsTty: false,
      writeOut(text) {
        cap.out += text;
      },
      writeErr(text) {
        cap.err += text;
      },
    },
  };
  return cap;
}

const losslessOpenAI = JSON.stringify({
  messages: [
    { role: "system", content: "Be nice." },
    { role: "user", content: "Hi" },
  ],
});

const imageOpenAI = JSON.stringify({
  messages: [
    {
      role: "user",
      content: [{ type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } }],
    },
  ],
});

describe("cli convert command", () => {
  it("converts from stdin and prints the payload to stdout, nothing to stderr when clean", () => {
    const cap = fakeIo(losslessOpenAI);
    const code = runConvertCommand(
      { args: [], flags: { from: "openai", to: "anthropic" } },
      cap.io,
    );
    expect(code).toBe(0);
    const parsed = JSON.parse(cap.out) as { system?: string };
    expect(parsed.system).toBe("Be nice.");
    // Zero notes -> no "no issues found" line, stderr stays empty.
    expect(cap.err).toBe("");
  });

  it("--json emits { output, notes } together on stdout", () => {
    const cap = fakeIo(losslessOpenAI);
    const code = runConvertCommand(
      { args: [], flags: { from: "openai", to: "gemini", json: true } },
      cap.io,
    );
    expect(code).toBe(0);
    const parsed = JSON.parse(cap.out) as { output: unknown; notes: unknown[] };
    expect(parsed).toHaveProperty("output");
    expect(parsed).toHaveProperty("notes");
    expect(Array.isArray(parsed.notes)).toBe(true);
  });

  it("--strict exits 1 when a conversion has a loss, 0 otherwise", () => {
    const lossy = fakeIo(imageOpenAI);
    const lossyCode = runConvertCommand(
      { args: [], flags: { from: "openai", to: "anthropic", strict: true, "text-only": true } },
      lossy.io,
    );
    expect(lossyCode).toBe(1);

    const clean = fakeIo(losslessOpenAI);
    const cleanCode = runConvertCommand(
      { args: [], flags: { from: "openai", to: "anthropic", strict: true } },
      clean.io,
    );
    expect(cleanCode).toBe(0);
  });

  it("rejects an unknown provider with exit 1 and a one-line error", () => {
    const cap = fakeIo(losslessOpenAI);
    const code = runConvertCommand({ args: [], flags: { from: "openai", to: "mistral" } }, cap.io);
    expect(code).toBe(1);
    expect(cap.err).toContain("--to must be one of");
    expect(cap.out).toBe("");
  });
});
