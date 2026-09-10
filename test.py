"""
Tests a new, more specific hypothesis than the earlier "fresh call"
theory (which was directly refuted by test_fresh_call_penalty.py's
results): that inserting the knowledge-base "RELEVANT KNOWLEDGE"
system message MID-ARRAY - between the last assistant reply and the
current user message, rather than appended at the end like every
other message - is what actually causes the slowdown. This matches
the exact, confirmed pattern in two separate real call logs: both
times, the first turn where a knowledge injection appeared showed
prompt-eval roughly doubling, disproportionate to the token growth.

This is the same FAMILY of issue as the trim penalty (a mid-array
structural change, not pure growth) but a different, more specific
trigger - worth testing directly rather than assuming it's the same
mechanism.

Three conditions, all reaching the SAME final token count for a fair
comparison:
  A. NORMAL - a knowledge injection message inserted mid-array, exactly
     like the real app does it
  B. APPENDED - the same knowledge content, but appended at the END
     of the array instead (like a normal new message), not inserted
     mid-array
  C. NEVER INSERTED - no knowledge injection at all, just the
     conversation growing normally to the same final size

Usage:
    python test_knowledge_injection_penalty.py
"""
import time
from ollama import Client

from app.llm.prompts import SYSTEM_PROMPT
from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT

MODEL = "gemma3:4b"
COMBINED_SYSTEM_PROMPT = f"{SYSTEM_PROMPT.strip()}\n\n{EMPLOYEE_ID_PROMPT.strip()}"

client = Client()

KNOWLEDGE_INJECTION = (
    "RELEVANT KNOWLEDGE (reference only — use to inform your answer, "
    "do not read it aloud or copy its wording):\n"
    "## Software\nID card designing, ODBC/database connectivity, visitor "
    "management, access-control management, BMS-related management "
    "solutions.\n## Notes\nAttendance/access-control integrations may use "
    "RFID, smart cards, biometrics, or other technology depending on the "
    "requirement — never claim a specific biometric product, model, or "
    "accuracy figure unless separately verified."
)

EARLIER_TURNS = [
    ("Yeah, sure, go ahead.", "Okay. Could you tell me a bit more about the scale of your employee ID program?"),
    ("We are able to provide services in Kashmir.", "That's great to hear. Our PAN India service extends to Kashmir. Would you like me to connect you with a specialist?"),
    ("Where do you operate from?", "I'm calling from Smart Chip Tree — offices in Delhi, Noida, Bangalore and Pune, HQ in Delhi."),
]

FINAL_TURN = "Like simple PVC cards, and lanyards, and also the printing. And you mentioned that the printing services are free. Is that right?"


def warmup():
    client.chat(
        model=MODEL,
        messages=[{"role": "system", "content": COMBINED_SYSTEM_PROMPT}, {"role": "user", "content": "warmup"}],
        options={"num_predict": 5},
    )


def build_base_messages():
    messages = [{"role": "system", "content": COMBINED_SYSTEM_PROMPT}]
    for user_turn, assistant_turn in EARLIER_TURNS:
        messages.append({"role": "user", "content": user_turn})
        messages.append({"role": "assistant", "content": assistant_turn})
    return messages


def condition_a_mid_array_insertion():
    """Matches the real app exactly: knowledge injected as its own
    system message, positioned right before the final user message."""
    messages = build_base_messages()
    messages.append({"role": "system", "content": KNOWLEDGE_INJECTION})
    messages.append({"role": "user", "content": FINAL_TURN})
    response = client.chat(model=MODEL, messages=messages, options={"num_predict": 40})
    return response.prompt_eval_duration / 1e6, response.prompt_eval_count


def condition_b_appended_content():
    """Same knowledge content, same final token count, but folded
    into the user message itself rather than inserted as a separate
    message mid-array — tests whether the MID-ARRAY POSITION is what
    matters, not just the extra content."""
    messages = build_base_messages()
    combined_final = f"{FINAL_TURN}\n\n[{KNOWLEDGE_INJECTION}]"
    messages.append({"role": "user", "content": combined_final})
    response = client.chat(model=MODEL, messages=messages, options={"num_predict": 40})
    return response.prompt_eval_duration / 1e6, response.prompt_eval_count


def condition_c_no_injection():
    """No knowledge injection at all — just the plain final turn,
    for a clean baseline at a smaller token count."""
    messages = build_base_messages()
    messages.append({"role": "user", "content": FINAL_TURN})
    response = client.chat(model=MODEL, messages=messages, options={"num_predict": 40})
    return response.prompt_eval_duration / 1e6, response.prompt_eval_count


def main():
    print("Warming up model...")
    warmup()
    print()

    print("=" * 70)
    print("CONDITION A: knowledge injected MID-ARRAY (matches real app)")
    print("=" * 70)
    pe_ms, tokens = condition_a_mid_array_insertion()
    print(f"tokens={tokens}, prompt_eval={pe_ms:.1f}ms")

    print()
    print("=" * 70)
    print("CONDITION B: same knowledge content, folded into final message")
    print("=" * 70)
    pe_ms, tokens = condition_b_appended_content()
    print(f"tokens={tokens}, prompt_eval={pe_ms:.1f}ms")

    print()
    print("=" * 70)
    print("CONDITION C: no knowledge injection at all (smaller, baseline)")
    print("=" * 70)
    pe_ms, tokens = condition_c_no_injection()
    print(f"tokens={tokens}, prompt_eval={pe_ms:.1f}ms")

    print()
    print("=" * 70)
    print("HOW TO READ THIS:")
    print("- If A is meaningfully slower than B despite similar token")
    print("  counts, the MID-ARRAY INSERTION POSITION itself is the")
    print("  real cause — a genuine, fixable finding (e.g. append")
    print("  knowledge content differently, or fold it into the user")
    print("  message instead of inserting a separate system message).")
    print("- If A and B are similar (both slow, or both fast), the")
    print("  extra CONTENT/token count is what matters, not position -")
    print("  a much less useful, less actionable finding.")
    print("- C establishes what 'normal, no injection' costs at a")
    print("  comparable shape for context.")
    print("=" * 70)


if __name__ == "__main__":
    main()