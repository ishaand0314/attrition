/**
 * Validate + repair IR-level invariant violations, ONCE, between parse and
 * serialize. Provider-agnostic: it fixes the conversation so that all three
 * serializers can assume a well-formed IR, instead of each deciding
 * independently and producing three different behaviors for the same bad input.
 *
 * The only invariant it currently enforces: every tool result must answer a
 * tool call that exists in the conversation.
 *
 * Principle: when a conversion cannot be both faithful and valid, prefer valid
 * and honest over faithful and invented. So an orphan tool result is DROPPED
 * with a `loss` note. We never synthesize a fake preceding tool call to make it
 * validate — that fabricates assistant output the model never produced, an
 * invisible corruption strictly worse than dropping content.
 *
 * A tool CALL with no matching result is NOT a violation — it is the normal
 * mid-agent-loop state (the tool hasn't run yet) and every format accepts it.
 */

import { type Conversation, type Message, isToolCall, isToolResult } from "./message.js";
import type { NoteCollector } from "./notes.js";

/**
 * Mutates `conversation` in place: drops orphan tool results, and then drops any
 * message left with zero parts (an empty message that Anthropic would reject).
 * Records notes on `notes`.
 */
export function validate(conversation: Conversation, notes: NoteCollector): void {
  const callIds = new Set<string>();
  for (const msg of conversation.messages) {
    for (const part of msg.parts) {
      if (isToolCall(part)) callIds.add(part.id);
    }
  }

  const kept: Message[] = [];
  conversation.messages.forEach((msg, mi) => {
    const survivingParts = msg.parts.filter((part, pi) => {
      if (isToolResult(part) && !callIds.has(part.id)) {
        notes.loss(
          `Dropped tool result with no matching call (id: "${part.id}")`,
          `messages[${mi}].parts[${pi}]`,
        );
        return false;
      }
      return true;
    });

    if (survivingParts.length === 0) {
      // The whole message emptied out. If it started non-empty, dropping it is a
      // second-order effect worth its own note (Anthropic rejects empty messages).
      if (msg.parts.length > 0) {
        notes.warning(
          "Dropped a message left empty after removing an orphan tool result",
          `messages[${mi}]`,
        );
      }
      return;
    }

    msg.parts = survivingParts;
    kept.push(msg);
  });

  conversation.messages = kept;
}
