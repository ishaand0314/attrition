/**
 * A tiny zero-dependency CLI router.
 *
 * ── HAND-SYNCED WITH LABKIT ──────────────────────────────────────────────
 * This is the same router shipped in labkit's @labkit/core (core/src/cli.ts).
 * There is NO shared package between the repos, so it is COPIED here and kept
 * in sync by hand, to give every 7-in-7 tool an identical CLI UX (--json,
 * --help, consistent errors). If you change routing behavior, mirror it in
 * labkit, and vice versa.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Each tool registers named commands; this handles routing, `--help`, `--json`,
 * and unknown-command errors uniformly. Deliberately minimal — if a tool
 * outgrows it, reach for a real parser.
 */

export interface CommandContext {
  /** Positional args after the command name. */
  readonly args: string[];
  /** Parsed --flags (e.g. --json, --from openai). */
  readonly flags: Record<string, string | boolean>;
}

export interface Command {
  readonly name: string;
  readonly summary: string;
  run(ctx: CommandContext): Promise<void> | void;
}

export interface CliOptions {
  readonly name: string;
  readonly description: string;
  readonly commands: readonly Command[];
  /**
   * Flags that never take a value (e.g. `json`, `strict`). Listed here, they
   * parse as boolean `true` and do NOT swallow the following token. `json` and
   * `help` are always treated as boolean.
   */
  readonly booleanFlags?: readonly string[];
}

const ALWAYS_BOOLEAN = ["json", "help"] as const;

function parseFlags(
  tokens: string[],
  booleanFlags: readonly string[] = [],
): {
  args: string[];
  flags: Record<string, string | boolean>;
} {
  const booleans = new Set<string>([...ALWAYS_BOOLEAN, ...booleanFlags]);
  const args: string[] = [];
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (tok?.startsWith("--")) {
      const key = tok.slice(2);
      const next = tokens[i + 1];
      if (!booleans.has(key) && next !== undefined && !next.startsWith("--")) {
        flags[key] = next;
        i++;
      } else {
        flags[key] = true;
      }
    } else if (tok !== undefined) {
      args.push(tok);
    }
  }
  return { args, flags };
}

function printHelp(opts: CliOptions): void {
  console.log(`${opts.name}: ${opts.description}\n`);
  console.log("Commands:");
  for (const cmd of opts.commands) {
    console.log(`  ${cmd.name.padEnd(16)} ${cmd.summary}`);
  }
  console.log("\nFlags:");
  console.log("  --json           Machine-readable output");
  console.log("  --help           Show this help");
}

/** Entry point. Call from each tool's `cli.ts` with `run(opts, process.argv.slice(2))`. */
export async function run(opts: CliOptions, argv: string[]): Promise<void> {
  const [commandName, ...rest] = argv;
  if (!commandName || commandName === "--help" || commandName === "help") {
    printHelp(opts);
    return;
  }
  const command = opts.commands.find((c) => c.name === commandName);
  if (!command) {
    console.error(`Unknown command: ${commandName}\n`);
    printHelp(opts);
    process.exitCode = 1;
    return;
  }
  await command.run(parseFlags(rest, opts.booleanFlags));
}
