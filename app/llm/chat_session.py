from app.llm.ollama_client import LLM
from app.llm.prompts import SYSTEM_PROMPT, OPENING_MESSAGE
from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT
from app.knowledge_base import KnowledgeBase
from app.conversation.controller import ConversationController
import time
import re
import json
import uuid
import datetime
from pathlib import Path

TRANSCRIPT_DIR = Path("transcripts")
TRANSCRIPT_DIR.mkdir(exist_ok=True)

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


def _offers_specialist(reply: str) -> bool:
    """
    Detect whether Spark's reply is offering to connect the customer with
    a specialist. Exact-phrase matching against SPECIALIST_PHRASES is
    fragile — a generative model paraphrases constantly ("connect you
    with one", "have someone reach out", "loop in one of our specialists")
    and silently misses the fixed list, which stalls the whole scheduling
    flow (the app never learns an offer was made, so it can't recognize
    the customer's "yes" as accepting it). Falling back to a keyword
    heuristic catches the paraphrases the fixed list misses.
    """
    r = reply.lower()
    if any(phrase in r for phrase in SPECIALIST_PHRASES):
        return True
    return "specialist" in r and "?" in reply


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


# Filler words/affirmations people drop in front of a spoken answer
# ("uhh yeah 6pm", "um so 6pm", "uh Rahul"). Stripped from name,
# organization, and callback-time answers — narrow enough contexts that
# being aggressive here is safe.
_FILLER_WORDS = {
    "uh", "um", "umm", "uhh", "uhm", "erm", "ah", "hmm", "hm",
    "yeah", "yep", "yup", "yes", "sure", "okay", "ok", "well",
    "so", "alright", "right",
}


def _clean_spoken_text(text: str) -> str:
    words = text.strip().split()
    cleaned = [w for w in words if w.strip(".,!?").lower() not in _FILLER_WORDS]
    cleaned_text = " ".join(cleaned).strip()
    return cleaned_text if cleaned_text else text.strip()


# Safety net for the callback-time question specifically: people
# sometimes volunteer their name unprompted ("around 5pm, i am rahul")
# even when it was already asked for separately. If a self-identification
# phrase is found, pull the name out and strip it from the text so it
# doesn't end up inside the spoken-back time.
_SELF_ID_RE = re.compile(
    r"\b(?:i am|i'm|this is|my name is|its|it's)\s+([a-zA-Z]+)\b\.?,?",
    re.IGNORECASE,
)


def _extract_and_strip_name(text: str):
    match = _SELF_ID_RE.search(text)
    if not match:
        return text, None
    name = match.group(1).strip()
    cleaned = (text[:match.start()] + text[match.end():]).strip(" ,.-")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return (cleaned or text), name


class ChatSession:

    def __init__(self):

        self.waiting_for_callback_confirmation = False

        self.llm = LLM()
        self.knowledge = KnowledgeBase()
        self.call_started = False
        self.waiting_for_name = False
        self.waiting_for_organization = False
        self.waiting_for_callback_time = False
        self.controller = ConversationController()

        self.customer_name = None
        self.customer_organization = None

        # Counts substantive (knowledge-search) exchanges so we can nudge
        # the model harder to wrap up once enough info has been gathered.
        # gemma3:4b weighs recent context far more than instructions
        # buried in a long, distant system prompt — a small reminder
        # placed next to the latest message steers it far more reliably
        # than the same instruction sitting only in the static prefix.
        self.exchange_count = 0

        # Full, untrimmed record of the call for later data extraction —
        # separate from self.messages, which is deliberately trimmed for
        # latency and doesn't include the fast-path replies (rejection,
        # busy, scheduling, etc). Written to disk incrementally so a
        # crash mid-call doesn't lose the transcript.
        self.call_id = uuid.uuid4().hex
        self.transcript = []
        self._transcript_path = TRANSCRIPT_DIR / f"{self.call_id}.jsonl"

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

    def _log(self, role, content):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        self.transcript.append(entry)
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _reply(self, text):
        """Log an assistant reply to the transcript, then return it."""
        self._log("assistant", text)
        return text

    def save_summary(self, extra=None):
        """
        Write a single consolidated JSON for this call (transcript plus
        any scheduling/metadata you pass in via `extra`, e.g. callback
        time or lead info from self.controller). Call this when the call
        ends — on rejection, after a callback time is confirmed, or when
        your driver loop detects the user hung up / typed exit.
        """
        summary = {
            "call_id": self.call_id,
            "started": self.transcript[0]["timestamp"] if self.transcript else None,
            "customer_name": self.customer_name,
            "customer_organization": self.customer_organization,
            "transcript": self.transcript,
        }
        if extra:
            summary.update(extra)
        path = TRANSCRIPT_DIR / f"{self.call_id}_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path

    def send(self, user_message):

        user_message = user_message.strip()
        clean = _normalize(user_message)
        total_start = time.perf_counter()

        # Empty input
        if not user_message:
            return "Sorry, I didn't catch that."

        self._log("user", user_message)

        # Opening Message
        if not self.call_started:
            self.call_started = True
            return self._reply(OPENING_MESSAGE)

        if any(phrase in clean for phrase in {
            "what is this about",
            "whats this about",
            "what is this regarding",
            "whats this regarding",
            "why are you calling",
            "what do you offer",
            "what do you guys do",
            "what is smart chip"}):
            return self._reply(
                "We provide solutions for employee attendance, access control, "
                "RFID tracking, ID cards and customized software and hardware integration. "
                "For a limited time, we are also offering complimentary services for "
                "ID card printer servicing, card designing and related software services. "
                "I wanted to see if any of these areas are relevant to your business."
            )

        # Rejection
        if clean in REJECTION_PHRASES:
            self.controller.reject()
            return self._reply("No problem at all. Thanks for your time. Have a great day.")

        # Busy
        if clean in BUSY_PHRASES or "busy" in clean:
            self.waiting_for_callback_confirmation = True
            self.controller.busy()
            return self._reply("No problem. Would another time be better?")

        # Callback response — now reuses SPECIALIST_YES_PHRASES /
        # REJECTION_PHRASES instead of a separate hand-typed set that was
        # missing "absolutely" / "of course" (those used to fall through
        # to a full, slower LLM call instead of the instant fast path).
        if self.waiting_for_callback_confirmation:

            if clean in SPECIALIST_YES_PHRASES:
                self.waiting_for_callback_confirmation = False
                self.waiting_for_name = True
                self.controller.start_scheduling()
                return self._reply("Great. May I get your name?")

            if clean in REJECTION_PHRASES:
                self.waiting_for_callback_confirmation = False
                self.controller.reject()
                return self._reply("No problem, thank you for your time!")

        # Name and organization are asked as two SEPARATE questions,
        # deterministically, right after the customer agrees to a
        # specialist. Trying to parse both out of one combined answer
        # ("iifl india" — is that a name or a company?) is inherently
        # ambiguous; asking one at a time removes the ambiguity entirely.
        if self.waiting_for_name:
            self.waiting_for_name = False
            self.customer_name = _clean_spoken_text(user_message)
            self.waiting_for_organization = True
            return self._reply("Thanks. And which organization are you calling from?")

        if self.waiting_for_organization:
            self.waiting_for_organization = False
            self.customer_organization = _clean_spoken_text(user_message)
            self.controller.set_contact_details(self.customer_name, self.customer_organization)
            self.waiting_for_callback_time = True
            return self._reply("Got it. What time would you prefer for the callback?")

        # Customer directly requests a callback (skips the confirmation
        # question) — still routes through name/organization first so
        # every path to scheduling collects the same details.
        if clean in SCHEDULING_PHRASES:
            self.waiting_for_name = True
            self.controller.start_scheduling()
            return self._reply("Certainly. May I get your name?")

        # Callback time. Two cleanup passes: first pull out a name if the
        # customer volunteers one unprompted here anyway ("5pm, i am
        # rahul"), then strip filler words from what's left so "uhh yeah
        # 6pm" doesn't get spoken back verbatim. Raw text is still
        # preserved as-typed in the transcript regardless.
        if self.waiting_for_callback_time:
            self.waiting_for_callback_time = False

            remaining_text, extracted_name = _extract_and_strip_name(user_message)
            if extracted_name and not self.customer_name:
                self.customer_name = extracted_name.title()
                self.controller.set_contact_details(
                    self.customer_name, self.customer_organization
                )

            callback_time = _clean_spoken_text(remaining_text)
            self.controller.set_callback_time(callback_time)
            self.controller.close()

            name_part = f"{self.customer_name}, " if self.customer_name else ""
            return self._reply(
                f"Perfect, {name_part}our specialist will contact you at "
                f"{callback_time}. Thank you for your time, have a great day!"
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
        if _offers_specialist(reply):
            self.waiting_for_callback_confirmation = True
            self.controller.specialist_requested()

        # Store assistant response
        self.messages.append({"role": "assistant", "content": reply})

        # Keep conversation memory small
        if len(self.messages) > 5:
            self.messages = [self.messages[0], *self.messages[-4:]]

        if DEBUG_TIMING:
            print(f"[TIMING] Total response: {time.perf_counter() - total_start:.2f}s")

        return self._reply(reply) 