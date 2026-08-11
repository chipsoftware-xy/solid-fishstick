from app.llm.ollama_client import LLM
from app.llm.prompts import SYSTEM_PROMPT, OPENING_MESSAGE
from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT
from app.knowledge_base import KnowledgeBase
from app.conversation.controller import ConversationController
import time
import re

# Flip to False in production — the json/char-count logging below has a
# small but real per-turn cost (serializing the whole message list).
DEBUG_TIMING = True

REJECTION_PHRASES = {
    "no",
    "nope",
    "not interested",
    "not really",
    "no thanks",
}

BUSY_PHRASES = {
    "i am busy",
    "i'm busy",
    "i am busy right now",
    "i'm busy right now",
    "busy",
}

SPECIALIST_PHRASES = {
    "connect you with one of our specialists",
    "connect you with a specialist",
    "connect you to a specialist",
    "speak with one of our specialists",
    "speak to one of our specialists",
    "talk to one of our specialists",
    "speak with a specialist",
    "talk to a specialist",
    "specialist to discuss",
    "specialist call",
    "schedule a call",
    "arrange a call",
}

SPECIALIST_YES_PHRASES = {
    "yes",
    "yeah",
    "yep",
    "sure",
    "okay",
    "ok",
    "fine",
    "alright",
    "yes sure",
    "yeah sure",
    "yes please",
    "sure thing",
    "absolutely",
    "of course",
}

SCHEDULING_PHRASES = {
    "call me tomorrow",
    "call tomorrow",
    "contact me tomorrow",
    "tomorrow is fine",
    "tomorrow works",
    "call me later",
    "contact me later",
    "call me another time",
}

# Low-information replies like "yes"/"ok" carry no new topic for the
# knowledge search to match against. Without gating, a vague query like
# this can retrieve an unrelated top-1 chunk (e.g. old script text),
# which a small model may echo back verbatim. Skip retrieval for these
# and let the model continue from conversation history alone.
GENERIC_CONTINUATIONS = {
    "yes",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "fine",
    "alright",
    "yes please",
    "sounds good",
    "that works",
    "sure thing",
}

# Strip punctuation before matching so "No." / "Sure!" still hit the fast
# paths instead of falling through to a full (slow) LLM call.
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", text.strip().lower())


class ChatSession:

    def __init__(self):

        self.waiting_for_callback_confirmation = False

        self.llm = LLM()
        self.knowledge = KnowledgeBase()
        self.call_started = False
        self.waiting_for_callback_time = False
        self.waiting_for_specialist_details = False
        self.controller = ConversationController()

        # Counts substantive (knowledge-search) exchanges so we can nudge
        # the model harder to wrap up once enough info has been gathered.
        # gemma3:4b weighs recent context far more than instructions
        # buried in a long, distant system prompt — a small reminder
        # placed next to the latest message steers it far more reliably
        # than the same instruction sitting only in the static prefix.
        self.exchange_count = 0

        # SYSTEM_PROMPT + EMPLOYEE_ID_PROMPT are merged ONCE here, into a
        # single static system message that never changes for the life of
        # the session. Previously the full EMPLOYEE_ID_PROMPT text (plus
        # extra restated instructions) was rebuilt and reinserted on every
        # turn, which doubled prompt size and prevented Ollama from
        # reusing its KV cache across turns. Keeping this block byte-
        # identical every call lets the (large, fixed) instruction prefix
        # get cached instead of reprocessed each time.
        combined_system_prompt = (
            f"{SYSTEM_PROMPT.strip()}\n\n{EMPLOYEE_ID_PROMPT.strip()}"
        )

        self.messages = [
            {
                "role": "system",
                "content": combined_system_prompt
            }
        ]

    def send(self, user_message):

        user_message = user_message.strip()
        clean = _normalize(user_message)
        total_start = time.perf_counter()

        # Empty input
        if not user_message:
            return "Sorry, I didn't catch that."

        # Opening Message
        if not self.call_started:
            self.call_started = True
            return OPENING_MESSAGE

        if any(phrase in clean for phrase in {
            "what is this about",
            "whats this about",
            "what is this regarding",
            "whats this regarding",
            "why are you calling",
            "what do you offer",
            "what do you guys do",
            "what is smart chip"}):
            return (
                "We provide solutions for employee attendance, access control, "
                "RFID tracking, ID cards and customized software and hardware integration. "
                "For a limited time, we are also offering complimentary services for "
                "ID card printer servicing, card designing and related software services. "
                "I wanted to see if any of these areas are relevant to your business."
            )

        # Rejection
        if clean in REJECTION_PHRASES:
            return "No problem at all. Thanks for your time. Have a great day."

        # Busy
        if clean in BUSY_PHRASES or "busy" in clean:
            self.waiting_for_callback_confirmation = True
            return "No problem. Would another time be better?"

        # Callback response — now reuses SPECIALIST_YES_PHRASES /
        # REJECTION_PHRASES instead of a separate hand-typed set that was
        # missing "absolutely" / "of course" (those used to fall through
        # to a full, slower LLM call instead of the instant fast path).
        if self.waiting_for_callback_confirmation:

            if clean in SPECIALIST_YES_PHRASES:
                self.waiting_for_callback_confirmation = False
                self.waiting_for_callback_time = True
                self.controller.start_scheduling()
                return "Okay, what time would be suitable for you?"

            if clean in REJECTION_PHRASES:
                self.waiting_for_callback_confirmation = False
                return "No problem, thank you for your time!"

        # Customer directly requests a callback
        if clean in SCHEDULING_PHRASES:
            self.waiting_for_callback_time = True
            return "Certainly. What time would be convenient for you?"

        # Callback time
        if self.waiting_for_callback_time:
            self.waiting_for_callback_time = False
            self.controller.set_callback_time(user_message)
            return (
                f"Perfect, our specialist will contact you at {user_message}. "
                "Thank you for your time, have a great day!"
            )

        # Add customer message to conversation memory
        self.messages.append({"role": "user", "content": user_message})
        self.exchange_count += 1

        # Skip retrieval for low-signal replies — nothing meaningful to
        # search for, and an irrelevant top-1 match risks getting echoed
        # back verbatim by the model (see GENERIC_CONTINUATIONS above).
        if clean in GENERIC_CONTINUATIONS:
            knowledge_context = ""
        else:
            start = time.perf_counter()
            knowledge_results = self.knowledge.search(user_message, n_results=1)
            if DEBUG_TIMING:
                print(f"[TIMING] Knowledge search: {time.perf_counter() - start:.2f}s")
            knowledge_context = "\n\n".join(knowledge_results)

        # Short, cheap reminder placed right next to the latest user
        # message. The full rules already live in the static system
        # prompt, but a weak model needs the critical ones repeated close
        # to generation time to reliably follow them — especially
        # "don't repeat the opening," which it otherwise tends to lift
        # verbatim from earlier context on vague turns.
        if self.exchange_count >= 2:
            stage_note = (
                "the customer has now described what they need. Do not ask "
                "further clarifying questions (size, design, quantity, "
                "color, etc). Acknowledge you can help and ask if they'd "
                "like to be connected with a specialist."
            )
        else:
            stage_note = (
                "if the requirement is already clear, don't ask more "
                "questions — acknowledge it and offer a specialist."
            )

        reminder = (
            "REMINDER: continue naturally from the conversation so far. "
            "Do not repeat the call opening and do not reintroduce "
            "yourself — that already happened. Do not state you're an AI "
            f"unless directly asked. And {stage_note}"
        )

        knowledge_block = (
            f"RELEVANT KNOWLEDGE (reference only — use to inform your "
            f"answer, do not read it aloud or copy its wording):\n"
            f"{knowledge_context}\n\n{reminder}"
            if knowledge_context else reminder
        )

        llm_messages = list(self.messages)
        llm_messages.insert(-1, {
            "role": "system",
            "content": knowledge_block
        })

        if DEBUG_TIMING:
            prompt_chars = sum(len(m["content"]) for m in llm_messages)
            print(f"[TIMING] Prompt size: {prompt_chars} characters")

        start = time.perf_counter()
        reply = self.llm.generate(llm_messages)
        if DEBUG_TIMING:
            print(f"[TIMING] LLM generation: {time.perf_counter() - start:.2f}s")

        # Detect when Spark is offering a specialist call
        if any(phrase in reply.lower() for phrase in SPECIALIST_PHRASES):
            self.waiting_for_callback_confirmation = True
            self.controller.specialist_requested()

        # Store assistant response
        self.messages.append({"role": "assistant", "content": reply})

        # Keep conversation memory small
        if len(self.messages) > 5:
            self.messages = [self.messages[0], *self.messages[-4:]]

        if DEBUG_TIMING:
            print(f"[TIMING] Total response: {time.perf_counter() - total_start:.2f}s")

        return reply