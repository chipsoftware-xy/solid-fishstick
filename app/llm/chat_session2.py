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


# Strips a leading self-identification phrase before storing a name
# answer, so "my name is Rahul" / "I'm Rahul" / "it's Rahul" all end up
# stored as just "Rahul" instead of the whole sentence.
_NAME_PREFIX_RE = re.compile(
    r"^\s*(?:my name is|the name is|i am|i'm|this is|it'?s|call me)\s+",
    re.IGNORECASE,
)


def _extract_name_answer(text: str) -> str:
    stripped = _NAME_PREFIX_RE.sub("", text.strip()).rstrip(".,!?").strip()
    cleaned = _clean_spoken_text(stripped) if stripped else _clean_spoken_text(text)
    return cleaned or text.strip()


# Same idea for organization answers: "it's Acme Corp" / "I'm calling
# from Acme" / "we're with Acme" -> just "Acme Corp".
_ORG_PREFIX_RE = re.compile(
    r"^\s*(?:i'?m(?:\s+calling)?\s+from|we'?re(?:\s+calling)?\s+from|"
    r"calling from|i work for|i'?m with|we'?re with|it'?s|this is|from)\s+",
    re.IGNORECASE,
)


def _extract_org_answer(text: str) -> str:
    stripped = _ORG_PREFIX_RE.sub("", text.strip()).rstrip(".,!?").strip()
    cleaned = _clean_spoken_text(stripped) if stripped else _clean_spoken_text(text)
    return cleaned or text.strip()


# Loose check that a callback-time answer actually looks like a time,
# rather than blindly accepting whatever text came out of STT (garbled
# audio was previously being read back verbatim as the scheduled time,
# e.g. "contact you at Try this experience.."). Deliberately permissive
# — this only needs to catch clearly-not-a-time garbage, not validate
# real time formats precisely.
_TIME_SIGNAL_RE = re.compile(
    r"\d|am\b|pm\b|a\.m|p\.m|morning|afternoon|evening|tonight|noon|"
    r"midnight|today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|o'?clock",
    re.IGNORECASE,
)


def _looks_like_time(text: str) -> bool:
    return bool(_TIME_SIGNAL_RE.search(text))


# Splits a stream of text deltas into complete sentences as soon as
# terminal punctuation appears, so each sentence can be handed to TTS the
# moment it's ready instead of waiting for the whole reply. Any leftover
# partial text is flushed as a final "sentence" once the stream ends, so
# nothing generated is ever dropped even if the model doesn't end its
# reply on clean punctuation.
_SENTENCE_END_RE = re.compile(r"[.!?](\s|$)")


def _stream_to_sentences(delta_iter):
    buffer = ""
    for delta in delta_iter:
        buffer += delta
        while True:
            match = _SENTENCE_END_RE.search(buffer)
            if not match:
                break
            split_at = match.end()
            sentence = buffer[:split_at].strip()
            if sentence:
                yield sentence
            buffer = buffer[split_at:]
    if buffer.strip():
        yield buffer.strip()


# Printer brands worth proactively connecting to the wholesale-ribbon
# offer. This fact already lives in the static system prompt (see
# EMPLOYEE_ID_PROMPT's FOCUS list), but a distant, static mention isn't
# enough for gemma3:4b to reliably act on — same pattern as the
# specialist-push and no-repeat-opening reminders. Detecting the brand
# mention in code and reinforcing it right next to the latest message
# makes the connection deterministic instead of hoping the model infers it.
_PRINTER_BRAND_ALIASES = {
    "zebra": "Zebra",
    "evolis": "Evolis",
    "hid fargo": "HID Fargo",
    "fargo": "HID Fargo",
    "datacard": "DataCard",
    "data card": "DataCard",
    "idp": "IDP",
    "retransfer": "retransfer",
}


def _mentioned_printer_brand(user_message: str):
    text = user_message.lower()
    for alias, canonical in _PRINTER_BRAND_ALIASES.items():
        if alias in text:
            return canonical
    return None


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
        self._time_retry_count = 0
        self._pending_callback_time = None

        # Brand mentioned in the CURRENT turn (set in _build_llm_messages,
        # consumed/cleared in _maybe_add_ribbon_fallback). Brands already
        # offered ribbons for are tracked so we don't repeat the same
        # upsell every time the customer references the same brand again.
        self._pending_brand = None
        self.brands_offered_ribbons_for = set()

        # Counts substantive (knowledge-search) exchanges so we can nudge
        # the model harder to wrap up once enough info has been gathered.
        self.exchange_count = 0

        # Full, untrimmed record of the call for later data extraction —
        # separate from self.messages, which is deliberately trimmed for
        # latency. Written to disk incrementally so a crash mid-call
        # doesn't lose the transcript.
        self.call_id = uuid.uuid4().hex
        self.transcript = []
        self._transcript_path = TRANSCRIPT_DIR / f"{self.call_id}.jsonl"

        # SYSTEM_PROMPT + EMPLOYEE_ID_PROMPT merged ONCE into a static
        # system message that never changes for the life of the session,
        # so Ollama can reuse its KV cache across turns instead of
        # reprocessing the full instructions every time.
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
        Write a single consolidated JSON for this call. Call this when
        the call ends — on rejection, after a callback time is
        confirmed, or when your driver loop detects hangup/exit.
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

    # ------------------------------------------------------------------
    # Deterministic fast path — every branch here is a canned reply, no
    # LLM call involved. Shared by both send() and send_streaming() so
    # this logic only exists once. Returns the reply string if this turn
    # was fully handled here, or None if it needs to go to the LLM.
    # ------------------------------------------------------------------
    def _try_fast_path(self, user_message, clean):

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

        # Callback confirmation response
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

            if _looks_like_time(user_message):
                # Safety net: the model may have asked "what time works
                # for you" itself in the same breath as the specialist
                # offer, despite being told not to — a known failure mode
                # even with the reminder above. Rather than losing this
                # answer to a generic LLM turn (which previously caused
                # the call to end without ever collecting name/org/time),
                # treat it as an implicit yes and capture the time now.
                # Name + organization are still collected normally; the
                # separate time question just gets skipped since we
                # already have it.
                self.waiting_for_callback_confirmation = False
                self._pending_callback_time = _clean_spoken_text(user_message)
                self.waiting_for_name = True
                self.controller.start_scheduling()
                return self._reply("Great. May I get your name?")

        # Name and organization — two separate deterministic questions.
        if self.waiting_for_name:
            self.waiting_for_name = False
            self.customer_name = _extract_name_answer(user_message)
            self.waiting_for_organization = True
            return self._reply("Thanks. And which organization are you calling from?")

        if self.waiting_for_organization:
            self.waiting_for_organization = False
            self.customer_organization = _extract_org_answer(user_message)
            self.controller.set_contact_details(self.customer_name, self.customer_organization)

            if self._pending_callback_time:
                # Time was already captured during the confirmation step
                # (see the safety net above) — skip asking again.
                callback_time = self._pending_callback_time
                self._pending_callback_time = None
                self.controller.set_callback_time(callback_time)
                self.controller.close()
                name_part = f"{self.customer_name}, " if self.customer_name else ""
                return self._reply(
                    f"Perfect, {name_part}our specialist will contact you "
                    f"at {callback_time}. Thank you for your time, have a "
                    f"great day!"
                )

            self.waiting_for_callback_time = True
            return self._reply("Got it. What time would you prefer for the callback?")

        # Customer directly requests a callback, skipping confirmation.
        if clean in SCHEDULING_PHRASES:
            self.waiting_for_name = True
            self.controller.start_scheduling()
            return self._reply("Certainly. May I get your name?")

        # Callback time.
        if self.waiting_for_callback_time:

            remaining_text, extracted_name = _extract_and_strip_name(user_message)
            if extracted_name and not self.customer_name:
                self.customer_name = extracted_name.title()
                self.controller.set_contact_details(
                    self.customer_name, self.customer_organization
                )

            callback_time = _clean_spoken_text(remaining_text)

            if not _looks_like_time(callback_time):
                self._time_retry_count += 1
                if self._time_retry_count >= 2:
                    # Cap retries so a customer isn't trapped in a loop
                    # if STT keeps failing on this turn — give up
                    # gracefully rather than repeating the question
                    # indefinitely.
                    self.waiting_for_callback_time = False
                    self.controller.set_callback_time(None)
                    self.controller.close()
                    return self._reply(
                        "No problem — our specialist will reach out to "
                        "find a time that works. Thank you for your "
                        "time, have a great day!"
                    )
                return self._reply(
                    "Sorry, I didn't catch a valid time — could you say "
                    "that again? For example, 'tomorrow at 3pm'."
                )

            self.waiting_for_callback_time = False
            self.controller.set_callback_time(callback_time)
            self.controller.close()

            name_part = f"{self.customer_name}, " if self.customer_name else ""
            return self._reply(
                f"Perfect, {name_part}our specialist will contact you at "
                f"{callback_time}. Thank you for your time, have a great day!"
            )

        return None

    # ------------------------------------------------------------------
    # Shared LLM-turn setup: appends to history, retrieves knowledge,
    # builds the reminder block. Used by both the blocking and streaming
    # LLM paths so this logic only exists once.
    # ------------------------------------------------------------------
    def _build_llm_messages(self, user_message, clean):
        self.messages.append({"role": "user", "content": user_message})
        self.exchange_count += 1

        if clean in GENERIC_CONTINUATIONS:
            knowledge_context = ""
        else:
            start = time.perf_counter()
            # n_results=2 rather than 1: this knowledge base only chunks
            # into 2 pieces total (see "Knowledge base ready (2 chunks)"
            # in your logs), so retrieving just 1 meant Spark only ever
            # saw HALF the company knowledge on any given turn, and which
            # half was unpredictable. At this corpus size there's no real
            # benefit to similarity filtering — just always include both.
            # If employee_id.md grows significantly later, this should
            # be revisited (bump back down + rely on the distance
            # threshold in KnowledgeBase.search).
            knowledge_results = self.knowledge.search(user_message, n_results=2)
            if DEBUG_TIMING:
                print(f"[TIMING] Knowledge search: {time.perf_counter() - start:.2f}s")
            knowledge_context = "\n\n".join(knowledge_results)

        if self.exchange_count >= 2:
            stage_note = (
                "the customer has now described what they need. Do not ask "
                "further clarifying questions (size, design, quantity, "
                "color, etc). Acknowledge you can help and ask ONLY if "
                "they'd like to be connected with a specialist — do not "
                "ask what time works for them, do not suggest a day or "
                "time yourself. The app asks for a time separately, only "
                "after they say yes."
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
            f"unless directly asked. Keep this reply short — one or two "
            f"sentences, under 35 words total; do not stack three or "
            f"more sentences even if you have more you could say. And "
            f"{stage_note}"
        )

        brand = _mentioned_printer_brand(user_message)
        self._pending_brand = brand
        if brand:
            reminder += (
                f" The customer just mentioned {brand}. Proactively "
                f"mention that Smart Chip Tree offers wholesale rates on "
                f"{brand} printer ribbons, even though they didn't ask "
                f"about ribbons directly."
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

        return llm_messages

    def _maybe_add_ribbon_fallback(self, reply, on_sentence=None):
        """
        Guarantees the wholesale-ribbon upsell actually happens when a
        printer brand is mentioned, instead of hoping the model acts on
        the reminder (it doesn't reliably — see the DataCard turn where
        it didn't, then the Zebra turn where it did). If the model
        already brought it up on its own, this is a no-op. Only fires
        once per brand per call, so referencing the same brand again
        later doesn't repeat the same upsell every time.

        `on_sentence`, if given (streaming path), is called with the
        fallback sentence so it actually gets synthesized and played —
        by this point every sentence the LLM produced has already been
        spoken, so just appending text to `reply` would leave it in the
        transcript without the customer ever hearing it.
        """
        brand = self._pending_brand
        self._pending_brand = None

        if not brand or brand in self.brands_offered_ribbons_for:
            return reply

        self.brands_offered_ribbons_for.add(brand)

        if "wholesale" in reply.lower() or "ribbon" in reply.lower():
            return reply

        fallback_sentence = (
            f"By the way, we also offer wholesale rates on {brand} "
            f"printer ribbons."
        )

        if on_sentence:
            on_sentence(fallback_sentence)

        reply = reply.rstrip()
        if reply and reply[-1] not in ".!?":
            reply += "."
        return f"{reply} {fallback_sentence}"

    def _finalize_llm_reply(self, reply):
        if _offers_specialist(reply):
            self.waiting_for_callback_confirmation = True
            self.controller.specialist_requested()

        self.messages.append({"role": "assistant", "content": reply})

        if len(self.messages) > 5:
            self.messages = [self.messages[0], *self.messages[-4:]]

        return self._reply(reply)

    def send(self, user_message):
        """Blocking version — waits for the full reply before returning."""
        user_message = user_message.strip()
        total_start = time.perf_counter()

        if not user_message:
            return "Sorry, I didn't catch that."

        self._log("user", user_message)
        clean = _normalize(user_message)

        fast_reply = self._try_fast_path(user_message, clean)
        if fast_reply is not None:
            return fast_reply

        llm_messages = self._build_llm_messages(user_message, clean)

        start = time.perf_counter()
        reply = self.llm.generate(llm_messages)
        if DEBUG_TIMING:
            print(f"[TIMING] LLM generation: {time.perf_counter() - start:.2f}s")

        reply = self._maybe_add_ribbon_fallback(reply)
        result = self._finalize_llm_reply(reply)

        if DEBUG_TIMING:
            print(f"[TIMING] Total response: {time.perf_counter() - total_start:.2f}s")

        return result

    def send_streaming(self, user_message, on_sentence):
        """
        Streaming version for live calls. `on_sentence(text)` is called
        once per complete sentence as soon as it's ready — wire this to
        your TTS/playback so the caller starts hearing Spark speak after
        the first sentence instead of waiting for the whole reply.

        Fast-path (deterministic) replies call on_sentence exactly once
        with the full canned reply, since there's nothing to stream —
        those are already instant. Returns the full reply text either
        way, same as send().
        """
        user_message = user_message.strip()
        total_start = time.perf_counter()

        if not user_message:
            reply = "Sorry, I didn't catch that."
            on_sentence(reply)
            return reply

        self._log("user", user_message)
        clean = _normalize(user_message)

        fast_reply = self._try_fast_path(user_message, clean)
        if fast_reply is not None:
            on_sentence(fast_reply)
            return fast_reply

        llm_messages = self._build_llm_messages(user_message, clean)

        start = time.perf_counter()
        full_sentences = []
        first_sentence_at = None

        for sentence in _stream_to_sentences(self.llm.generate_stream(llm_messages)):
            if first_sentence_at is None:
                first_sentence_at = time.perf_counter()
                if DEBUG_TIMING:
                    print(f"[TIMING] Time to first sentence: {first_sentence_at - start:.2f}s")
            full_sentences.append(sentence)
            on_sentence(sentence)

        if DEBUG_TIMING:
            print(f"[TIMING] LLM generation (full): {time.perf_counter() - start:.2f}s")

        reply = " ".join(full_sentences)
        reply = self._maybe_add_ribbon_fallback(reply, on_sentence=on_sentence)
        result = self._finalize_llm_reply(reply)

        if DEBUG_TIMING:
            print(f"[TIMING] Total response: {time.perf_counter() - total_start:.2f}s")

        return result