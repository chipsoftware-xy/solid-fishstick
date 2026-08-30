"""
Direct test of whether Ollama's KV cache actually helps OUR usage
pattern — a growing, multi-turn conversation with NEW content each
turn, not a repeated identical request. That distinction matters: the
original bug report's own single-turn identical-repeat test and its
multi-turn growing-conversation test showed DIFFERENT results, and our
actual usage is the growing-conversation kind. Testing on a tiny
repeated prompt (as an earlier test did) doesn't tell us anything
about the scenario we actually care about.

Uses the REAL system prompt (same one chat_session.py sends) for
realism, and a 5-turn conversation with different content each turn —
matching our actual production pattern, not a synthetic shortcut.

Run this directly from the project root:
    python test_ollama_cache.py

Tests BOTH /api/chat (what chat_session.py currently uses) and
/api/generate + context (the potential migration target) against the
IDENTICAL multi-turn scenario, back to back, so they're directly
comparable on your exact Ollama version and hardware — a real
measurement instead of reasoning about someone else's bug report.
"""
import time
from ollama import Client

from app.llm.prompts import SYSTEM_PROMPT
from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT

MODEL = "gemma3:4b"  # change if your model tag differs

COMBINED_SYSTEM_PROMPT = f"{SYSTEM_PROMPT.strip()}\n\n{EMPLOYEE_ID_PROMPT.strip()}"

# Five turns of genuinely different content, styled after real customer
# messages seen in testing — NOT identical repeated messages, since
# that's a different (and apparently different-behaving) caching
# scenario than what we actually need.
TURNS = [
    "Yeah sure, what is the software thing you mentioned?",
    "We are actually an organization and we currently need like 50 PVC cards.",
    "But also tell me this, what is the pricing structure?",
    "I mean I don't mind, but I'm already using a DataCard printer, what would that look like?",
    "Okay sure, that works for me.",
]

client = Client()


def run_chat_test():
    print("\n" + "=" * 70)
    print("TEST 1: /api/chat — what chat_session.py currently uses")
    print("=" * 70)
    messages = [{"role": "system", "content": COMBINED_SYSTEM_PROMPT}]
    prev_pe_count = 0
    for i, turn in enumerate(TURNS, 1):
        messages.append({"role": "user", "content": turn})
        start = time.perf_counter()
        response = client.chat(model=MODEL, messages=messages, options={"num_predict": 40})
        elapsed = time.perf_counter() - start
        pe_count = response.prompt_eval_count
        pe_ms = response.prompt_eval_duration / 1e6
        new_tokens = pe_count - prev_pe_count
        ms_per_new_token = pe_ms / new_tokens if new_tokens else float("nan")
        print(
            f"Turn {i}: pe_count={pe_count} (+{new_tokens} new), "
            f"pe_ms={pe_ms:.1f}ms, ms/new_token={ms_per_new_token:.2f}, "
            f"wall={elapsed:.2f}s"
        )
        prev_pe_count = pe_count
        messages.append({"role": "assistant", "content": response.message.content})


def run_generate_test():
    print("\n" + "=" * 70)
    print("TEST 2: /api/generate + context — potential migration target")
    print("=" * 70)
    context = None
    prev_pe_count = 0
    for i, turn in enumerate(TURNS, 1):
        kwargs = {"model": MODEL, "prompt": turn, "options": {"num_predict": 40}}
        if i == 1:
            kwargs["system"] = COMBINED_SYSTEM_PROMPT
        if context is not None:
            kwargs["context"] = context
        start = time.perf_counter()
        response = client.generate(**kwargs)
        elapsed = time.perf_counter() - start
        pe_count = response.prompt_eval_count
        pe_ms = response.prompt_eval_duration / 1e6
        new_tokens = pe_count - prev_pe_count
        ms_per_new_token = pe_ms / new_tokens if new_tokens else float("nan")
        print(
            f"Turn {i}: pe_count={pe_count} (+{new_tokens} new), "
            f"pe_ms={pe_ms:.1f}ms, ms/new_token={ms_per_new_token:.2f}, "
            f"wall={elapsed:.2f}s"
        )
        prev_pe_count = pe_count
        context = response.context


if __name__ == "__main__":
    run_chat_test()
    run_generate_test()
    print("\n" + "=" * 70)
    print("HOW TO READ THIS:")
    print("- If a method's cache is NOT helping: pe_ms grows roughly in")
    print("  step with pe_count every turn, so ms/new_token stays roughly")
    print("  CONSTANT turn over turn (this is what the original bug")
    print("  report's own multi-turn test showed for /api/chat on CPU).")
    print("- If a method's cache IS helping: ms/new_token stays LOW and")
    print("  roughly flat after turn 1, regardless of how large pe_count")
    print("  (the total, cumulative context) has grown.")
    print("- Compare the two ms/new_token columns directly against each")
    print("  other — that comparison is the actual answer, not either")
    print("  test read in isolation.")
    print("=" * 70)