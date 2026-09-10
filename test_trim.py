"""
Fixed version - v1 had two real bugs:
1. Never included the "reminder" system message chat_session.py adds
   every turn, so message count grew too slowly (2/turn instead of 3)
   and never actually crossed the real >15 trim threshold at all.
2. Computed ms/tok using TOTAL prompt_eval_count instead of the
   INCREMENT since the previous call, making the rate look
   artificially tiny and meaningless.

This version matches chat_session.py's real per-turn message shape
exactly, includes enough turns to definitely trigger a trim, uses
the correct incremental-token calculation, and explicitly isolates
the first real post-trim call for a second, same-shape comparison.

Usage:
    python test_trim_repeat_v2.py
"""
import time
from ollama import Client

from app.llm.prompts import SYSTEM_PROMPT
from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT

MODEL = "gemma3:4b"
COMBINED_SYSTEM_PROMPT = f"{SYSTEM_PROMPT.strip()}\n\n{EMPLOYEE_ID_PROMPT.strip()}"
REMINDER = (
    "REMINDER: continue naturally from the conversation so far. Do not "
    "repeat the call opening and do not reintroduce yourself."
)

client = Client()

# 10 turns guarantees at least one full trim cycle with the real
# 3-messages-per-turn shape (trim fires once messages > 15).
TURNS = [
    "Yeah sure, what is the software thing you mentioned?",
    "We are actually an organization and we currently need like 50 PVC cards.",
    "But also tell me this, what is the pricing structure?",
    "I mean I don't mind, but I'm already using a DataCard printer, what would that look like?",
    "Okay sure, that works for me.",
    "Do you also deal with biometric stuff?",
    "Where are you based?",
    "What's the biggest weakness of your company?",
    "Would you personally buy your own product?",
    "What is your phone number?",
]


def call_and_report(label, messages, prev_tokens):
    start = time.perf_counter()
    response = client.chat(model=MODEL, messages=messages, options={"num_predict": 40})
    wall = time.perf_counter() - start
    pe_ms = response.prompt_eval_duration / 1e6
    tokens = response.prompt_eval_count
    new_tokens = tokens - prev_tokens if prev_tokens is not None else tokens
    ms_per_new = pe_ms / new_tokens if new_tokens else float("nan")
    print(
        f"{label}: total_tokens={tokens} (+{new_tokens} new), "
        f"prompt_eval={pe_ms:.1f}ms, ms/new_tok={ms_per_new:.2f}, wall={wall:.2f}s"
    )
    return response, tokens


def main():
    messages = [{"role": "system", "content": COMBINED_SYSTEM_PROMPT}]
    prev_tokens = None
    trimmed_once = False
    post_trim_messages = None

    for i, turn in enumerate(TURNS, 1):
        messages.append({"role": "system", "content": REMINDER})
        messages.append({"role": "user", "content": turn})
        response, prev_tokens = call_and_report(f"Turn {i} ({len(messages)} msgs)", messages, prev_tokens)
        messages.append({"role": "assistant", "content": response.message.content})

        if len(messages) > 15:
            messages = [messages[0], *messages[-4:]]
            print(f"  [TRIMMED to {len(messages)} messages]")
            if not trimmed_once:
                trimmed_once = True
                # Save this exact shape so we can test it a second time
                # right after the real post-trim call below.
                post_trim_messages = [dict(m) for m in messages]

    if not trimmed_once:
        print("\nTrim never fired - increase TURNS further and re-run.")
        return

    print("\n--- Testing the SAME post-trim shape a SECOND time ---")
    print("(reusing the exact saved shape from right after the first real trim)")
    second_messages = post_trim_messages + [{"role": "user", "content": "Where are you based?"}]
    call_and_report("Second call, same post-trim shape", second_messages, None)

    print("\nHOW TO READ THIS:")
    print("- Find the turn immediately after '[TRIMMED...]' above - that's")
    print("  the real first post-trim call. Compare ITS prompt_eval time")
    print("  to the 'Second call, same post-trim shape' result just above.")
    print("- If the second one is much faster, the post-trim cost is a")
    print("  one-time penalty - pre-warming could genuinely help.")
    print("- If they're similar, every call at a freshly-trimmed shape")
    print("  pays the cost - pre-warming would not help.")


if __name__ == "__main__":
    main()