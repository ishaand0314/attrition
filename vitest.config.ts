import { defineConfig } from "vitest/config";

/**
 * Tests import directly from `src` with relative paths, so `pnpm test` runs
 * against TypeScript source with no build step and always reflects the latest
 * code. This is a single-package repo, so — unlike labkit's monorepo — there is
 * no workspace alias to configure here.
 */
export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
  },
});
