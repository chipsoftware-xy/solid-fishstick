from app.llm.ollama_client import LLM
from app.llm.prompts import SYSTEM_PROMPT, OPENING_MESSAGE
from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT
from app.knowledge_base import KnowledgeBase
from app.conversation.controller import ConversationController
from app.config import settings
import time
import re
import difflib
import json
import uuid
import datetime
import threading
from pathlib import Path

_STALL_THRESHOLD_SECONDS = settings["llm"]["stall_threshold_seconds"]
_STALL_PHRASE = settings["llm"]["stall_phrase"]

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


# Detects when the MODEL's own reply sounds like a natural closing
# (mimicking the closing examples in SYSTEM_PROMPT — "I'll have our team
# take it forward. Thanks for your time, have a great day.") — without
# this, that reply sounds like a real goodbye but nothing marks the call
# as actually ended, so the customer's next reply (even a plain "you
# too, thanks") gets treated as a fresh turn and Spark restarts pitching
# from scratch instead of winding the call down.
_CLOSING_PHRASES = {
    "have a great day",
    "thanks for your time",
    "take it forward",
}


def _sounds_like_closing(reply: str) -> bool:
    r = reply.lower()
    return any(phrase in r for phrase in _CLOSING_PHRASES)


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


def _contains_phrase(text: str, phrases: set) -> bool:
    """
    Word-boundary-aware "does this phrase appear anywhere in the text"
    check — catches natural combinations like "yeah sure of course"
    (contains both "yeah sure" and "of course" as phrases, but isn't
    equal to either), which exact-match membership (`clean in phrases`)
    misses entirely. Uses \\b boundaries rather than naive substring
    search so "yes" doesn't wrongly match inside "yesterday".
    """
    return any(re.search(rf"\b{re.escape(p)}\b", text) for p in phrases)


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
    if not _looks_like_name(name):
        # The "match" is likely a false positive ("I am pretty flexible"
        # matching "I am" + "pretty") — don't strip it out of the text
        # either, since that would corrupt whatever's left (e.g. leaving
        # "flexible, so..." as the base for time extraction instead of
        # the original full sentence).
        return text, None
    cleaned = (text[:match.start()] + text[match.end():]).strip(" ,.-")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return (cleaned or text), name


# Strips a leading self-identification phrase before storing a name
# answer, so "my name is Rahul" / "I'm Rahul" / "it's Rahul" all end up
# stored as just "Rahul" instead of the whole sentence. NOT anchored to
# the start (\b, not ^) — people constantly lead with a conversational
# preamble first ("Of course, my name is Rahul"), and an anchored match
# would silently fail to strip anything in that case, since the pattern
# never actually appears at position zero.
_NAME_PREFIX_RE = re.compile(
    r"\b(?:my name is|the name is|i am|i'm|this is|it'?s|call me)\s+",
    re.IGNORECASE,
)


# Truncates at the first sentence boundary or coordinating conjunction,
# so a rambling multi-clause answer doesn't all get captured as the
# name/org value. Word-boundary matched, so "Andy" isn't split on "and".
# The period must be followed by whitespace/end-of-string, not just any
# period — otherwise "10 a.m. that way..." gets chopped mid-abbreviation
# ("10 a" — the period inside "a.m." was being treated as a sentence
# end).
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[.!?](?=\s|$)|,?\s+\b(?:and|so|but)\b", re.IGNORECASE
)


def _first_clause(text: str) -> str:
    return _CLAUSE_BOUNDARY_RE.split(text, maxsplit=1)[0].strip()


def _extract_name_answer(text: str) -> str:
    clause = _first_clause(text.strip())
    match = _NAME_PREFIX_RE.search(clause)
    if match:
        stripped = clause[match.end():].rstrip(".,!?").strip()
    else:
        stripped = clause.rstrip(".,!?").strip()
    cleaned = _clean_spoken_text(stripped) if stripped else _clean_spoken_text(clause)
    return cleaned or text.strip()


# Same idea for organization answers: "it's Acme Corp" / "I'm calling
# from Acme" / "we're with Acme" -> just "Acme Corp". Also not anchored
# to the start, for the same reason as the name prefix above.
_ORG_PREFIX_RE = re.compile(
    r"\b(?:i'?m(?:\s+calling)?\s+from|we'?re(?:\s+calling)?\s+from|"
    r"calling from|i work for|i'?m with|we'?re with|with|it'?s|this is|from)\s+",
    re.IGNORECASE,
)

# Fallback for when the org appears mid-sentence with no recognized
# prefix at all — "with X" / "represents X" style phrasing.
_FROM_ANYWHERE_RE = re.compile(
    r"\bfrom\s+([A-Za-z0-9&.\-]+(?:\s+[A-Za-z0-9&.\-]+){0,3})",
    re.IGNORECASE,
)


def _extract_org_answer(text: str) -> str:
    clause = _first_clause(text.strip())

    match = _ORG_PREFIX_RE.search(clause)
    if match:
        stripped = clause[match.end():].rstrip(".,!?").strip()
        cleaned = _clean_spoken_text(stripped) if stripped else _clean_spoken_text(clause)
        return cleaned or text.strip()

    # No prefix matched — look for "from X" anywhere in the clause.
    from_match = _FROM_ANYWHERE_RE.search(clause)
    if from_match:
        return _clean_spoken_text(from_match.group(1).strip())

    cleaned = _clean_spoken_text(clause)
    return cleaned or text.strip()


def _llm_extract_field(llm, field_label, question_asked, raw_answer):
    """
    Uses the LLM itself to pull a single field (name/org/time) out of a
    conversational reply, instead of continuing to extend regex
    heuristics — those kept breaking on new phrasing patterns that
    real conversation kept producing: leading acknowledgment sentences
    ("Ah, sure. I'm calling from Acme..." — regex grabbed "Ah, sure" and
    threw away the actual company), declarative non-answers ("That
    would be great, if you could just let me know..." isn't a name at
    all but isn't a single bad word or a question either, so nothing
    caught it), casual phrases misread as self-identification ("I am
    pretty flexible"). This is exactly the kind of short, meaning-based
    extraction an LLM handles far more robustly than pattern matching.

    Kept deliberately narrow — a tiny, focused prompt, not the full
    conversation — so it stays fast. Only runs once per field per call
    (when the dedicated question is asked), not on every turn.

    Returns the extracted value, or None if nothing was found or the
    call failed.
    """
    messages = [
        {
            "role": "system",
            "content": (
                f"Extract ONLY the {field_label} from the customer's "
                f"reply below, given they were just asked: "
                f"\"{question_asked}\"\n"
                f"Reply with ONLY the extracted value — no explanation, "
                f"no extra words, no surrounding punctuation. If the "
                f"reply does not actually contain a {field_label} (for "
                f"example it's a question, an unrelated comment, or "
                f"just an acknowledgment like \"sure\" or \"okay\" with "
                f"nothing else), reply with exactly: NONE"
            ),
        },
        {"role": "user", "content": raw_answer},
    ]
    try:
        result = llm.generate(messages).strip().strip('"').strip()
    except Exception as e:
        print(f"[WARN] LLM field extraction failed, falling back: {e}")
        return None
    if not result or result.upper() == "NONE":
        return None
    return result


# Loose check that a callback-time answer actually looks like a time,
# rather than blindly accepting whatever text came out of STT (garbled
# audio was previously being read back verbatim as the scheduled time,
# e.g. "contact you at Try this experience.."). Deliberately permissive
# for genuine time phrasing, but NOT for bare digits — that was a real
# bug: a plain \d matched ANY digit anywhere in the sentence, so "we
# need around 5000 cards" (a quantity, not a time) satisfied it just
# because "5000" contains digits. Real time expressions have digits in
# a recognizable context — next to am/pm, in HH:MM form, or right after
# "at" — a bare 3+ digit number never does. Restricting to \d{1,2} in
# those specific contexts rejects quantities/prices/addresses while
# still catching "7pm", "3:30", "at 5", etc.
#
# Also no bare "am"/"pm" WORD match — "am" is an ordinary English word
# ("I am asking about..."), and \bam\b matched it regardless of
# context. Genuine time expressions always pair am/pm directly with a
# digit, which the patterns below already require.
_TIME_SIGNAL_RE = re.compile(
    r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)\b|"  # 7pm, 7:30pm, 7 a.m.
    r"\b\d{1,2}:\d{2}\b|"  # 7:30
    r"\bat\s+\d{1,2}\b|"  # "at 5", "at 3"
    r"morning|afternoon|evening|tonight|noon|"
    r"midnight|today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|o'?clock",
    re.IGNORECASE,
)


def _looks_like_time(text: str) -> bool:
    return bool(_TIME_SIGNAL_RE.search(text))


def _extract_time_phrase(text: str) -> str:
    """
    Extracts just the time-relevant portion of a longer sentence instead
    of using the whole thing verbatim — "thank you, I appreciate that,
    I'll be ready after 2pm then" should yield "after 2pm", not the
    entire sentence.

    Spans from the FIRST time-signal match to the LAST, not just a
    window around the first one — a sentence commonly has more than one
    signal ("tomorrow afternoon... maybe around 2pm"), and anchoring on
    only the first match risked capturing the vague reference
    ("tomorrow") while silently dropping the actually useful specific
    time ("2pm") if it appeared later in the sentence.

    Also truncates to the first clause first — trailing chatter after
    the time itself ("...would be ideal for me. Does that work for
    you?") shouldn't be included.

    Falls back to the fully cleaned text if no time signal is found
    (shouldn't happen if _looks_like_time already passed, but keeps
    this safe to call standalone).
    """
    clause = _first_clause(text)
    matches = list(_TIME_SIGNAL_RE.finditer(clause))
    if not matches:
        return _clean_spoken_text(text)

    words = clause.split()

    def word_index_at(char_pos):
        pos = 0
        for i, word in enumerate(words):
            pos += len(word) + 1
            if pos > char_pos:
                return i
        return len(words) - 1

    start_idx = word_index_at(matches[0].start())
    end_idx = word_index_at(max(matches[-1].end() - 1, matches[-1].start()))

    start_idx = max(0, start_idx - 1)
    end_idx = min(len(words) - 1, end_idx + 1)

    window = " ".join(words[start_idx:end_idx + 1]).strip(" ,.!?")
    return _clean_spoken_text(window)


# Words STT sometimes garbles a name into that clearly aren't names —
# catches cases like "Not" (as seen in testing: mis-heard audio ending up
# stored as the customer's literal name). Deliberately narrow: only
# rejects when the ENTIRE answer is a single one of these words, since a
# real name could legitimately contain "So" or "Ok" as a nickname/part
# of a longer name and we don't want to reject those.
_NOT_A_NAME = {
    "not", "no", "nope", "yes", "yeah", "yep", "what", "who", "sorry",
    "okay", "ok", "um", "uh", "hello", "hi", "sure",
    # Common adjectives/states/adverbs that follow "I'm" or "it's" in
    # ordinary speech but are never names — "I'm pretty flexible",
    # "it's fine", "I'm really busy". English is genuinely ambiguous
    # between "I'm [Name]" and "I'm [adjective]" here, so this list
    # can't be exhaustive — it catches the common offenders, not all.
    "pretty", "really", "very", "quite", "totally", "fine", "good",
    "great", "happy", "glad", "busy", "certain", "ready", "flexible",
    "open", "excited", "confident", "honestly", "just", "still",
    "also", "here", "there", "done", "afraid", "worried", "interested",
    "available", "free", "well",
}


def _looks_like_name(text: str) -> bool:
    words = text.strip().split()
    if not words:
        return False
    if len(words) == 1 and words[0].strip(".,!?").lower() in _NOT_A_NAME:
        return False
    return True


# Symmetric to _NOT_A_NAME, for the organization field — nothing
# previously validated this at all (a bare "One." got stored as the
# literal company name). Includes number words since a stray digit/count
# ("One", "1") is a common STT mis-hearing of something else entirely,
# not a plausible company name on its own.
_NOT_AN_ORG = _NOT_A_NAME | {
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
}


def _looks_like_org(text: str) -> bool:
    words = text.strip().split()
    if not words:
        return False
    if len(words) == 1 and words[0].strip(".,!?").lower() in _NOT_AN_ORG:
        return False
    return True


# Catches the customer answering a direct question (name/org) with
# ANOTHER question instead — e.g. "where's your office?" in response to
# "may I get your name?" — which previously got stored as if it were the
# actual answer (customer_name literally became "Where is your office").
# Must be checked against the RAW user_message, before any cleanup
# strips the trailing "?" or the leading question word.
_QUESTION_STARTERS = (
    "where", "what", "who", "why", "how", "when", "which",
    "can you", "do you", "are you", "is it", "could you", "will you",
)


def _looks_like_a_question(raw_text: str) -> bool:
    stripped = raw_text.strip()
    if stripped.endswith("?"):
        return True
    lowered = stripped.lower()
    return any(lowered.startswith(starter) for starter in _QUESTION_STARTERS)


# Catches the customer asking to have details/company info sent to them
# ("can you send me your details over this number", "text me the
# profile", "whatsapp me a brochure") — a "send" verb and a "content"
# noun within a short window of each other, rather than requiring exact
# phrasing. Reply must stay in FUTURE tense ("we'll send...") per
# SYSTEM_PROMPT's rule against claiming a message has already been sent
# — nothing sends anything yet; this only acknowledges the request.
_SEND_DETAILS_RE = re.compile(
    r"\b(?:send|share|text|whatsapp|email|forward|message)\b.{0,30}"
    r"\b(?:details?|info(?:rmation)?|profile|brochure|documents?|"
    r"catalog(?:ue)?|materials?)\b",
    re.IGNORECASE,
)


def _asks_to_send_details(raw_text: str) -> bool:
    return bool(_SEND_DETAILS_RE.search(raw_text))


# Lets the customer correct a mis-heard name/organization at ANY point
# in the call, not just during the dedicated question — e.g. "actually
# it's Alex, not Ali". Only checked once self.customer_name /
# self.customer_organization is already set (see the gating in
# _try_fast_path), so this never fires during the initial collection
# question, which is handled separately by _extract_name_answer.
_NAME_CORRECTION_RE = re.compile(
    r"\b(?:my name is|call me|it'?s|i'?m)\s+([a-zA-Z]+)",
    re.IGNORECASE,
)
_ORG_CORRECTION_RE = re.compile(
    r"\b(?:i'?m calling from|we'?re calling from|calling from|"
    r"my company is|my organization is|it'?s)\s+([A-Za-z0-9&.\- ]+?)"
    r"(?:[.,!?]|$)",
    re.IGNORECASE,
)


def _detect_name_correction(text: str):
    match = _NAME_CORRECTION_RE.search(text)
    return match.group(1).strip() if match else None


def _detect_org_correction(text: str):
    match = _ORG_CORRECTION_RE.search(text)
    return match.group(1).strip() if match else None


# Catches a BARE negation with no replacement — "my name isn't that" /
# "that's not my name" — as distinct from a correction ("it's Alex"),
# which _detect_name_correction already handles. A bare negation has
# nothing for that regex to extract (note: "isn't" also has no space
# after "is", so "my name is X" doesn't even match it as a prefix), so
# it was previously silently falling through to a generic LLM reply
# with the wrong name left in place uncorrected. Checked against
# `clean` (already lowercased/apostrophe-stripped by _normalize), not
# raw text.
_NAME_NEGATION_RE = re.compile(
    r"\bmy name (?:is ?not|isnt)\b|\bthats not my name\b|"
    r"\bthat is not my name\b|\bnot my name\b|\bwrong name\b"
)


def _is_name_negation_without_replacement(clean_text: str) -> bool:
    return bool(_NAME_NEGATION_RE.search(clean_text))


# STT frequently mangles short affirmations/rejections into nonsense
# ("Yashur", "Yashore" for "yeah sure") — an exact-match check sends
# these to a full ~15-20s LLM call for what should be an instant
# deterministic response, and it can take several tries before one
# happens to transcribe cleanly enough to match. Fuzzy-matching short
# inputs against the known target phrases catches most of these without
# a real fix to STT itself. Deliberately restricted to short inputs
# (<=3 words) to avoid ever mismatching a genuine, longer sentence.
def _fuzzy_match_short_phrase(clean_text: str, targets: set, threshold: float = 0.6):
    words = clean_text.split()
    if not words or len(words) > 3:
        return None
    best, best_ratio = None, 0.0
    for target in targets:
        ratio = difflib.SequenceMatcher(None, clean_text, target).ratio()
        if ratio > best_ratio:
            best_ratio, best = ratio, target
    return best if best_ratio >= threshold else None


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
    # Customers commonly say just "HID" without "Fargo" attached. Short
    # alias like this needs word-boundary matching (see below) — a
    # naive substring check would false-positive on "hidden", "forbid",
    # etc.
    "hid": "HID Fargo",
    "datacard": "DataCard",
    "data card": "DataCard",
    "idp": "IDP",
    "retransfer": "retransfer",
}


def _mentioned_printer_brand(user_message: str):
    text = user_message.lower()
    for alias, canonical in _PRINTER_BRAND_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return canonical
    return None


class ChatSession:

    def __init__(self, llm=None, knowledge=None):
        """
        `llm` and `knowledge` can be passed in as already-loaded
        instances — this is how repeated calls within the same running
        process (e.g. the web UI's Start button, reused across multiple
        calls) avoid reloading the model and knowledge base every time.
        If not given, fresh ones are created (matches the old
        behavior — fine for a single one-off call, e.g. from the CLI).
        """

        self.waiting_for_callback_confirmation = False

        self.llm = llm if llm is not None else LLM()
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase()
        self.call_started = False
        self.call_finished = False
        self.waiting_for_name = False
        self.waiting_for_organization = False
        self.waiting_for_callback_time = False
        self.controller = ConversationController()

        self.customer_name = None
        self.customer_organization = None
        self.customer_callback_time = None
        self._time_retry_count = 0
        self._name_retry_count = 0
        self._org_retry_count = 0
        self._pending_callback_time = None

        # Brand mentioned in the CURRENT turn (set in _build_llm_messages,
        # consumed/cleared in _maybe_add_ribbon_fallback). Brands already
        # offered ribbons for are tracked so we don't repeat the same
        # upsell every time the customer references the same brand again.
        self._pending_brand = None
        self.brands_offered_ribbons_for = set()
        self.specialist_offer_count = 0

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

        # Let the customer correct a mis-heard name/org at ANY point,
        # not just during the dedicated question. Gated on the field
        # already being set, so this never intercepts the initial
        # collection question itself (self.customer_name is still None
        # at that point, handled separately by _extract_name_answer).
        if self.customer_name and (
            "my name is" in clean
            or "call me" in clean
            or "actually" in clean
            or "correction" in clean
            or "not " in clean
        ):
            corrected = _detect_name_correction(user_message)
            if (
                corrected
                and _looks_like_name(corrected)
                and corrected.lower() != self.customer_name.lower()
            ):
                self.customer_name = corrected.title()
                self.controller.set_contact_details(
                    self.customer_name, self.customer_organization
                )
                return self._reply(
                    f"Got it — I've updated your name to {self.customer_name}."
                )
            if _is_name_negation_without_replacement(clean):
                # They rejected the stored name but didn't say what it
                # should be instead — ask, rather than silently doing
                # nothing and leaving the wrong name in place.
                self.customer_name = None
                self.waiting_for_name = True
                return self._reply("Sorry about that — what's your actual name?")

        if self.customer_organization and (
            "my organization is" in clean
            or "my company is" in clean
            or "calling from" in clean
            or "actually" in clean
            or "correction" in clean
        ):
            corrected_org = _detect_org_correction(user_message)
            if (
                corrected_org
                and _looks_like_org(corrected_org)
                and corrected_org.lower() != self.customer_organization.lower()
            ):
                self.customer_organization = corrected_org
                self.controller.set_contact_details(
                    self.customer_name, self.customer_organization
                )
                return self._reply(
                    f"Got it — I've updated your organization to "
                    f"{self.customer_organization}."
                )

        # Same idea for the callback time — e.g. after Spark has already
        # confirmed a time, "actually not that time, make it 7pm" or
        # "wrong time, change it to tomorrow morning" should update it,
        # not just get spoken back and forgotten. Gated on a time
        # already being confirmed AND the new text actually looking like
        # a time (see _looks_like_time), so "actually" alone mid-
        # conversation about something unrelated doesn't misfire.
        if self.customer_callback_time and (
            "actually" in clean
            or "not this time" in clean
            or "wrong time" in clean
            or "change the time" in clean
            or "correction" in clean
            or "instead" in clean
        ) and _looks_like_time(user_message):
            new_time = _extract_time_phrase(user_message)
            if new_time and new_time.lower() != self.customer_callback_time.lower():
                self.customer_callback_time = new_time
                self.controller.set_callback_time(new_time)
                return self._reply(
                    f"Got it — I've updated the callback time to {new_time}."
                )

        # Auto-end the call. controller.memory.conversation_ended is
        # already True once either close() (successful scheduling) or
        # reject() has run — meaning the customer already heard the
        # "Perfect... have a great day!" (or rejection) closing on the
        # PREVIOUS turn. Checked AFTER all three correction blocks
        # above, so a genuine correction attempt ("actually, my name
        # isn't right") still gets handled instead of being swallowed
        # by this — only a reply that doesn't match any correction
        # trigger (a plain "thank you", "bye", etc) reaches here. Give
        # one short deterministic reply and signal the driver loop
        # (main.py) to end the call, instead of listening indefinitely.
        if self.controller.memory.conversation_ended and not self.call_finished:
            self.call_finished = True
            return self._reply("Thank you, have a great day!")

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

        if _asks_to_send_details(user_message):
            return self._reply(
                "Yes, absolutely — we'll send over our company details "
                "and profile to this number after the call."
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

            # Contains-match first (catches natural phrasing like "yeah
            # sure of course" that exact-match membership misses
            # entirely, since it isn't equal to any single phrase in the
            # set); fuzzy match as a second fallback specifically for
            # garbled short STT output ("Yashur", "Yashore" for "yeah
            # sure") that doesn't contain a clean matching phrase at
            # all. Without either, each garbled/naturally-phrased
            # attempt costs a full ~15-20s LLM call for what should be
            # an instant response.
            if _contains_phrase(clean, SPECIALIST_YES_PHRASES) or _fuzzy_match_short_phrase(
                clean, SPECIALIST_YES_PHRASES
            ):
                self.waiting_for_callback_confirmation = False
                self.waiting_for_name = True
                self.controller.start_scheduling()
                return self._reply("Great. May I get your name?")

            if _contains_phrase(clean, REJECTION_PHRASES) or _fuzzy_match_short_phrase(
                clean, REJECTION_PHRASES
            ):
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
                self._pending_callback_time = _extract_time_phrase(user_message)
                self.waiting_for_name = True
                self.controller.start_scheduling()
                return self._reply("Great. May I get your name?")

            # None of the above matched — the customer didn't answer
            # yes/no/time, they've moved on to something else (a new
            # question, a new topic). Clear the flag rather than
            # leaving it stuck True indefinitely across future turns,
            # which was silently accumulating specialist_offer_count
            # every time the model happened to re-offer — treating a
            # customer asking three different good questions in a row
            # as if they'd ignored the same offer three times, and
            # cutting off an actively engaged conversation. A genuine
            # new question resets the "count" of ignored offers, since
            # it isn't one.
            self.waiting_for_callback_confirmation = False
            if _looks_like_a_question(user_message):
                self.specialist_offer_count = 0

        # Name and organization — two separate deterministic questions.
        if self.waiting_for_name:
            if _looks_like_a_question(user_message):
                self._name_retry_count += 1
                if self._name_retry_count >= 2:
                    self.waiting_for_name = False
                    self.waiting_for_organization = True
                    return self._reply(
                        "No worries — and which organization are you "
                        "calling from?"
                    )
                return self._reply(
                    "I'll get to that in just a moment — first, could I "
                    "get your name?"
                )

            name_candidate = _llm_extract_field(
                self.llm, "customer's name", "May I get your name?", user_message
            )
            if not name_candidate:
                # LLM call failed or found nothing — fall back to regex.
                name_candidate = _extract_name_answer(user_message)

            if not _looks_like_name(name_candidate):
                self._name_retry_count += 1
                if self._name_retry_count >= 2:
                    # Give up gracefully rather than looping — proceed
                    # without a name rather than trapping the customer.
                    self.waiting_for_name = False
                    self.waiting_for_organization = True
                    return self._reply(
                        "No worries — and which organization are you "
                        "calling from?"
                    )
                return self._reply(
                    "Sorry, I didn't catch your name — could you say "
                    "that again?"
                )

            self.waiting_for_name = False
            self.customer_name = name_candidate.title()
            self.waiting_for_organization = True
            return self._reply("Thanks. And which organization are you calling from?")

        if self.waiting_for_organization:
            if _looks_like_a_question(user_message):
                self._org_retry_count += 1
                if self._org_retry_count >= 2:
                    self.waiting_for_organization = False
                    self.waiting_for_callback_time = True
                    return self._reply(
                        "No worries — what time would you prefer for "
                        "the callback?"
                    )
                return self._reply(
                    "I'll get to that shortly — first, which "
                    "organization are you calling from?"
                )

            org_candidate = _llm_extract_field(
                self.llm,
                "organization or company name",
                "Which organization are you calling from?",
                user_message,
            )
            if not org_candidate:
                org_candidate = _extract_org_answer(user_message)

            if not _looks_like_org(org_candidate):
                self._org_retry_count += 1
                if self._org_retry_count >= 2:
                    self.waiting_for_organization = False
                    self.waiting_for_callback_time = True
                    return self._reply(
                        "No worries — what time would you prefer for "
                        "the callback?"
                    )
                return self._reply(
                    "Sorry, I didn't catch that — which organization "
                    "are you calling from?"
                )

            self.waiting_for_organization = False
            self.customer_organization = org_candidate
            self.controller.set_contact_details(self.customer_name, self.customer_organization)

            if self._pending_callback_time:
                # Time was already captured during the confirmation step
                # (see the safety net above) — skip asking again.
                callback_time = self._pending_callback_time
                self._pending_callback_time = None
                self.customer_callback_time = callback_time
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

            time_candidate = _llm_extract_field(
                self.llm,
                "callback time (day and/or time of day)",
                "What time would you prefer for the callback?",
                remaining_text,
            )
            if not time_candidate:
                time_candidate = _extract_time_phrase(remaining_text)

            if not _looks_like_time(time_candidate):
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
            callback_time = time_candidate
            self.customer_callback_time = callback_time
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
            self.specialist_offer_count += 1
            if self.specialist_offer_count >= 3:
                # Offered a specialist 3+ times without ever getting a
                # clear yes — the customer is very likely declining in
                # phrasing our matching doesn't cleanly catch ("we're
                # pretty set", "I'm good for now"). Stop re-pitching and
                # close gracefully instead of asking a 4th time.
                self.controller.reject()
                reply = (
                    "No problem at all — I'll leave you to it. Thanks "
                    "for your time, have a great day!"
                )
            else:
                self.waiting_for_callback_confirmation = True
                self.controller.specialist_requested()
        elif _sounds_like_closing(reply):
            # The model produced its own natural-sounding goodbye
            # without going through the deterministic scheduling/
            # rejection paths — mark the call as actually ended anyway,
            # so the customer's next reply (even a plain "you too,
            # thanks") triggers the auto-end flow instead of Spark
            # restarting the pitch from scratch.
            self.controller.close()

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

        # Background timer: if the real first sentence hasn't shown up
        # within the threshold, play a quick filler so the customer
        # isn't sitting in dead air on a slow generation. event.wait()
        # returns immediately (without playing anything) if the real
        # content arrives before the timeout, instead of a plain sleep
        # that can't be interrupted.
        first_sentence_ready = threading.Event()

        def _stall_filler():
            if not first_sentence_ready.wait(timeout=_STALL_THRESHOLD_SECONDS):
                self._log("assistant", _STALL_PHRASE)
                on_sentence(_STALL_PHRASE)

        stall_thread = threading.Thread(target=_stall_filler, daemon=True)
        stall_thread.start()

        for sentence in _stream_to_sentences(self.llm.generate_stream(llm_messages)):
            if first_sentence_at is None:
                first_sentence_at = time.perf_counter()
                if DEBUG_TIMING:
                    print(f"[TIMING] Time to first sentence: {first_sentence_at - start:.2f}s")
                first_sentence_ready.set()
                # If the filler already started playing, wait for it to
                # finish before playing real content — audio must never
                # overlap. If it hasn't started (real content beat the
                # threshold), this returns immediately.
                stall_thread.join()
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