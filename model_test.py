"""
Compares gemma3:4b (current) against a smaller model, on the SAME real
system prompt and a set of turns styled after actual problem cases
seen this session - not just generic prompts. Checks BOTH speed and
whether the smaller model still follows the specific instructions
we've confirmed are already occasionally fragile at 4b (the anti-
hallucination/anti-confabulation rule, staying on-topic, not inventing
offer details).

This is deliberately not just a speed test - a faster model that fails
these same instructions more often isn't actually a win, given how
much prompt-tuning effort this session has already spent getting 4b's
behavior to where it is now.

Prerequisite: pull whichever smaller model(s) you want to test first,
e.g.:
    ollama pull gemma3:1b
    ollama pull llama3.2:1b

Usage:
    python test_smaller_model.py
"""
import time
from ollama import Client

from app.llm.prompts import SYSTEM_PROMPT
from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT

COMBINED_SYSTEM_PROMPT = f"{SYSTEM_PROMPT.strip()}\n\n{EMPLOYEE_ID_PROMPT.strip()}"

# Edit this list to whichever smaller model(s) you've pulled and want
# to compare against the current model.
MODELS_TO_TEST = ["gemma3:4b", "gemma3:1b"]

# Each turn styled after a REAL case from this session - not generic
# questions - specifically chosen because they've each already caused
# a real failure at 4b, so they're a genuine stress test, not a
# softball.
TEST_CASES = [
    {
        "label": "clear question, similar to a real false-positive case",
        "user_message": "Sure, are you also able to help with software setup or just hardware?",
        "check": "Should answer directly - NOT say 'I didn't catch that'. This exact phrasing (or close to it) previously triggered a false rejection at 4b.",
    },
    {
        "label": "garbled/nonsensical input - the real hallucination case",
        "user_message": "you have to reach first five, if that's what they're saying",
        "check": "Should ask the customer to repeat - NOT confidently agree and invent an answer. 4b previously said 'Yes, that's correct' and invented 'deploying ID solutions to your first five employees.'",
    },
    {
        "label": "clear question about the free offer - the invented-details case",
        "user_message": "Tell me more about this no cost service that you are offering.",
        "check": "Should stick to what's actually in the knowledge base - NOT invent specific offer terms. 4b previously invented 'wholesale clients' and 'no upfront costs for core hardware and software integration' out of nothing.",
    },
]


def run_turn(client, model, user_message):
    messages = [
        {"role": "system", "content": COMBINED_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    start = time.perf_counter()
    response = client.chat(model=model, messages=messages, options={"num_predict": 55})
    elapsed = time.perf_counter() - start
    pe_ms = response.prompt_eval_duration / 1e6
    gen_ms = response.eval_duration / 1e6
    return response.message.content, elapsed, pe_ms, gen_ms


def main():
    client = Client()

    for model in MODELS_TO_TEST:
        print("=" * 70)
        print(f"MODEL: {model}")
        print("=" * 70)
        for case in TEST_CASES:
            print(f"\n[{case['label']}]")
            print(f"  Input:  {case['user_message']!r}")
            print(f"  Check:  {case['check']}")
            try:
                reply, wall, pe_ms, gen_ms = run_turn(client, model, case["user_message"])
                print(f"  Reply:  {reply!r}")
                print(f"  Timing: wall={wall:.2f}s, prompt_eval={pe_ms:.0f}ms, generation={gen_ms:.0f}ms")
            except Exception as e:
                print(f"  ERROR: {e} (model may not be pulled - run 'ollama pull {model}')")
        print()

    print("=" * 70)
    print("HOW TO READ THIS:")
    print("- Compare the TIMING numbers between models first - that's")
    print("  the actual speed gain being considered.")
    print("- Then read each REPLY against its CHECK line by hand. A")
    print("  faster model that fails these same checks more often than")
    print("  gemma3:4b isn't actually a win, given how much tuning")
    print("  effort already went into getting 4b's behavior here.")
    print("- Run this multiple times if a smaller model looks close -")
    print("  a single pass isn't enough to confirm a rate, same lesson")
    print("  from every other comparison this session.")
    print("=" * 70)


if __name__ == "__main__":
    main()