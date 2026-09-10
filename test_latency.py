"""
Tests whether the specific ACT of trimming (self.messages shrinking
discontinuously, as chat_session.py's history-trim does every ~4
turns) causes an anomalous prompt-eval spike - distinct from the
already-confirmed "no caching on a normally growing prompt" finding.

A prior analysis of a real call log spotted a real pattern: prompt-eval
spikes to ~13-14s roughly every 4 turns, correlating with trim events.
But its proposed mechanism ("cache invalidation forces reprocessing of
MORE tokens") doesn't fit the data - the spike turns actually had FEWER
tokens than the normal turns right before them, not more. This test
isolates the real variable: is it the SHRINKING itself that's
expensive, or would a fresh conversation at the same (post-trim) size
be equally slow, meaning something else is going on?

Two scenarios, same final prompt size, same model:
  A) Build a growing conversation up past the trim threshold, actually
     trim it (mimicking chat_session.py exactly), then time the next
     call.
  B) Start a brand new conversation directly at the post-trim size
     (system + 4 recent messages) - no trimming ever happened, no
     prior larger context existed.

If A is much slower than B, that confirms something about the
trim/shrink event itself. If A and B are similar, the spike isn't
about shrinking specifically - it's something else entirely (worth
re-investigating from scratch).

Usage:
    python test_trim_spike.py
"""
import time
from ollama import Client

from app.llm.prompts import SYSTEM_PROMPT
from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT

MODEL = "gemma3:4b"
COMBINED_SYSTEM_PROMPT = f"{SYSTEM_PROMPT.strip()}\n\n{EMPLOYEE_ID_PROMPT.strip()}"

client = Client()

# Enough distinct turns to grow past the real trim threshold (15
# messages) at least twice, so scenario A actually exercises trimming
# more than once, not just barely.
TURNS = [
    "Yeah sure, what is the software thing you mentioned?",
    "We are actually an organization and we currently need like 50 PVC cards.",
    "But also tell me this, what is the pricing structure?",
    "I mean I don't mind, but I'm already using a DataCard printer, what would that look like?",
    "Okay sure, that works for me.",
    "Do you also deal with biometric stuff?",
    "Where are you based?",
    "What's the biggest weakness of your company?",
]


def run_with_real_trim():
    print("=" * 70)
    print("SCENARIO A: growing conversation, trimmed exactly like chat_session.py")
    print("=" * 70)
    messages = [{"role": "system", "content": COMBINED_SYSTEM_PROMPT}]
    prev_size = None
    for i, turn in enumerate(TURNS, 1):
        messages.append({"role": "user", "content": turn})
        start = time.perf_counter()
        response = client.chat(model=MODEL, messages=messages, options={"num_predict": 40})
        elapsed = time.perf_counter() - start
        pe_ms = response.prompt_eval_duration / 1e6
        size_note = ""
        if prev_size is not None and response.prompt_eval_count < prev_size:
            size_note = "  <-- SHRANK (post-trim turn)"
        print(f"Turn {i}: tokens={response.prompt_eval_count}, pe_ms={pe_ms:.1f}, wall={elapsed:.2f}s{size_note}")
        prev_size = response.prompt_eval_count
        messages.append({"role": "assistant", "content": response.message.content})

        # Mimic chat_session.py's exact trim logic
        if len(messages) > 15:
            messages = [messages[0], *messages[-4:]]
            print(f"  [TRIMMED to {len(messages)} messages]")


def run_fresh_at_same_size():
    print("\n" + "=" * 70)
    print("SCENARIO B: fresh conversation, started directly at post-trim size")
    print("(system + 4 messages, no larger context ever existed)")
    print("=" * 70)
    # Reconstruct a plausible post-trim message list directly, matching
    # what scenario A's trim would produce - a fresh start, not a
    # shrink from something bigger.
    messages = [
        {"role": "system", "content": COMBINED_SYSTEM_PROMPT},
        {"role": "user", "content": TURNS[0]},
        {"role": "assistant", "content": "Sure, here's some information about that."},
        {"role": "user", "content": TURNS[1]},
        {"role": "assistant", "content": "Got it, thanks for sharing that."},
    ]
    messages.append({"role": "user", "content": "What's the biggest weakness of your company?"})
    start = time.perf_counter()
    response = client.chat(model=MODEL, messages=messages, options={"num_predict": 40})
    elapsed = time.perf_counter() - start
    pe_ms = response.prompt_eval_duration / 1e6
    print(f"Fresh call: tokens={response.prompt_eval_count}, pe_ms={pe_ms:.1f}, wall={elapsed:.2f}s")


if __name__ == "__main__":
    run_with_real_trim()
    run_fresh_at_same_size()
    print("\n" + "=" * 70)
    print("HOW TO READ THIS:")
    print("- Look at the pe_ms on turns marked '<-- SHRANK' in Scenario A.")
    print("- Compare that number directly to Scenario B's single result,")
    print("  which is a similar-sized prompt that never went through a")
    print("  shrink event.")
    print("- If the shrunk turn is dramatically slower than the fresh one,")
    print("  that confirms the SHRINK EVENT ITSELF is expensive, not just")
    print("  general no-caching cost - a real, specific bug in how")
    print("  trimming interacts with Ollama, worth fixing directly.")
    print("- If they're similar, the earlier spike-pattern theory needs")
    print("  to be reconsidered from scratch - something else is going on.")
    print("=" * 70)