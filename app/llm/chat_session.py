from app.llm.ollama_client import LLM
from app.llm.prompts import SYSTEM_PROMPT, OPENING_MESSAGE, OPENING_GREETING, OPENING_PITCH
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

import random

_STALL_THRESHOLD_SECONDS = settings["llm"]["stall_threshold_seconds"]
_STALL_PHRASES = [
    settings["llm"]["stall_phrase"],
    "One moment please.",
    "Let me think about that for a second.",
    "Give me just a moment.",
    "Okay, let me see.",
]

TRANSCRIPT_DIR = Path("transcripts")
TRANSCRIPT_DIR.mkdir(exist_ok=True)
DEBUG_TIMING = True
DEBUG_PROMPT_DUMP = False  

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
    "not now",
    "not a good time",
    "not the right time",
    "cant talk now",
    "cant talk right now",
    "kind of busy",
    "in a meeting",
    "in the middle of something",
    "can i call you back",
    "can you call me later",
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
    r = reply.lower()
    if any(phrase in r for phrase in SPECIALIST_PHRASES):
        return True
    return "specialist" in r and "?" in reply
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
    "perfect",
    "that's right",
    "that's correct",
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

GREETING_PHRASES = {"hello", "hi", "hey", "hello there", "hi there", "hey there"}

NO_RESPONSE_SENTINEL = "__NO_CUSTOMER_RESPONSE__"

GRATITUDE_PHRASES = {
    "thank you", "thanks", "thank you very much", "thanks a lot",
    "okay thank you", "ok thank you",
}
#clean
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", text.strip().lower())


def _contains_phrase(text: str, phrases: set) -> bool:
    return any(re.search(rf"\b{re.escape(p)}\b", text) for p in phrases)

_REPEAT_REQUEST_PHRASES = {
    "what did you say",
    "what was that",
    "can you repeat",
    "can you repeat that",
    "could you repeat",
    "could you repeat that",
    "could you say that again",
    "can you say that again",
    "say that again",
    "say it again",
    "one more time",
    "i didnt catch that",
    "i couldnt hear you",
    "i cant hear you",
    "i missed that",
    "youre breaking up",
    "you were breaking up",
    "you cut out",
    "come again",
}

_REPEAT_REQUEST_EXACT = {"what", "huh", "pardon", "sorry what", "im sorry what"}

_REPEAT_REQUEST_MAX_WORDS = 8


def _requests_repeat(clean: str) -> bool:
    if clean in _REPEAT_REQUEST_EXACT:
        return True
    if len(clean.split()) > _REPEAT_REQUEST_MAX_WORDS:
        return False
    return _contains_phrase(clean, _REPEAT_REQUEST_PHRASES)

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
        return text, None
    cleaned = (text[:match.start()] + text[match.end():]).strip(" ,.-")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return (cleaned or text), name
_NAME_PREFIX_RE = re.compile(
    r"\b(?:my name is|the name is|i am|i'm|this is|it'?s|call me)\s*",
    re.IGNORECASE,
)

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
        return _clean_spoken_text(stripped) if stripped else ""
    stripped = clause.rstrip(".,!?").strip()
    cleaned = _clean_spoken_text(stripped) if stripped else _clean_spoken_text(clause)
    return cleaned or text.strip()

_ORG_PREFIX_RE = re.compile(
    r"\b(?:i'?m(?:\s+calling)?\s+from|we'?re(?:\s+calling)?\s+from|"
    r"calling from|i work for|i'?m with|we'?re with|with|it'?s|this is|"
    r"my organization is|my company is|from)\s*",
    re.IGNORECASE,
)

_FROM_ANYWHERE_RE = re.compile(
    r"\bfrom\s+([A-Za-z0-9&.\-]+(?:\s+[A-Za-z0-9&.\-]+){0,3})",
    re.IGNORECASE,
)


def _extract_org_answer(text: str) -> str:
    clause = _first_clause(text.strip())

    match = _ORG_PREFIX_RE.search(clause)
    if match:
        stripped = clause[match.end():].rstrip(".,!?").strip()
        return _clean_spoken_text(stripped) if stripped else ""

    from_match = _FROM_ANYWHERE_RE.search(clause)
    if from_match:
        return _clean_spoken_text(from_match.group(1).strip())

    cleaned = _clean_spoken_text(clause)
    return cleaned or text.strip()


def _llm_classify_product_or_callback(llm, raw_answer):
    messages = [
        {
            "role": "system",
            "content": (
                "The customer was just asked: \"Would you like to know "
                "about a specific product or service, or would you "
                "prefer I connect you with a specialist for a "
                "callback?\"\n"
                "Classify their reply below as wanting a CALLBACK "
                "(connecting with a specialist, being called back, "
                "scheduling a call) or PRODUCT (asking about or "
                "wanting to know about a product, service, or "
                "anything else that isn't a callback request).\n"
                "Reply with ONLY one word: CALLBACK or PRODUCT."
            ),
        },
        {"role": "user", "content": raw_answer},
    ]
    try:
        result = llm.generate(messages).strip().upper()
    except Exception as e:
        print(f"[WARN] LLM product-or-callback classification failed, falling back: {e}")
        return None
    print(f"[CLASSIFIER-DEBUG] product-or-callback: input={raw_answer!r} -> raw output={result!r}")
    if "CALLBACK" in result:
        return True
    if "PRODUCT" in result:
        return False
    return None

def _llm_extract_field(llm, field_label, question_asked, raw_answer):
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


def _extracted_value_is_grounded(extracted: str, original_message: str) -> bool:
    extracted_words = set(_normalize(extracted).split())
    original_words = set(_normalize(original_message).split())
    return bool(extracted_words & original_words)


def _extract_time_phrase(text: str) -> str:
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

_NOT_A_NAME = {
    "not", "no", "nope", "yes", "yeah", "yep", "what", "who", "sorry",
    "okay", "ok", "um", "uh", "hello", "hi", "sure",
    "pretty", "really", "very", "quite", "totally", "fine", "good",
    "great", "happy", "glad", "busy", "certain", "ready", "flexible",
    "open", "excited", "confident", "honestly", "just", "still",
    "also", "here", "there", "done", "afraid", "worried", "interested",
    "available", "free", "well",
    "perfect", "awesome", "excellent", "wonderful", "cool", "nice",
    "bye", "goodbye", "thanks", "thank you",
    "next", "then", "now", "today", "tomorrow", "later", "soon",
    "week", "month", "morning", "afternoon", "evening",
    "an", "a", "the", "and", "or", "but", "to", "of", "in", "on",
    "at", "is", "was", "for", "with", "my", "your",
    "idiot", "stupid", "moron", "loser", "fool", "dumbass", "asshole",
    "fucker", "bastard", "bitch", "bitches", "jerk", "shithead",
}

_NAME_MAX_WORDS = 3


def _looks_like_name(text: str) -> bool:
    words = text.strip().split()
    if not words:
        return False
    if len(words) == 1:
        word = words[0].strip(".,!?").lower()
        if word in _NOT_A_NAME:
            return False
        if len(word) < 3:
            return False
    if len(words) > _NAME_MAX_WORDS:
        return False
    if _is_sparks_own_filler_phrase(text):
        return False
    return True

_NOT_AN_ORG = _NOT_A_NAME | {
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hello", "got", "what?","no",
}

_SPARKS_OWN_FILLER_PHRASES = {
    _normalize(p) for p in (
        "Got it.",
        "Got it, one moment.",
        "Okay, just a second.",
        "Alright, noted.",
        "Sure, one moment.",
        "Right, thank you.",
        "Noted, thanks.",
        "One second.",
        "Understood, thanks.",
        *_STALL_PHRASES,
    )
}


def _is_sparks_own_filler_phrase(text: str) -> bool:
    return _normalize(text) in _SPARKS_OWN_FILLER_PHRASES


def _looks_like_org(text: str) -> bool:
    words = text.strip().split()
    if not words:
        return False
    if len(words) == 1 and words[0].strip(".,!?").lower() in _NOT_AN_ORG:
        return False
    if _is_sparks_own_filler_phrase(text):
        return False
    if _looks_like_a_question(text):
        return False
    return True

_LOCATION_QUESTION_PATTERNS = (
    "where are you based",
    "where are you located",
    "where you are based",
    "where are you calling from",
    "where do you operate",
    "wheres your office",
    "where is your office",
    "which city are you",
    "which city you are",
)

SPARK_LOCATION_ANSWER = (
    "We have offices in Noida, Bangalore, and Pune, with our headquarters in Delhi."
    "We provide PAN India service and on-site support."
)


def _is_location_question(text: str) -> bool:
    lowered = _normalize(text)
    return any(pattern in lowered for pattern in _LOCATION_QUESTION_PATTERNS)

_QUESTION_STARTERS = (
    "where", "what", "who", "why", "how", "when", "which",
    "can you", "do you", "are you", "is it", "could you", "will you",
)

_NOT_ACTUALLY_A_QUESTION = {"hello", "hi", "hey", "hello there", "hey there"} | SPECIALIST_YES_PHRASES | REJECTION_PHRASES


def _looks_like_a_question(raw_text: str) -> bool:
    stripped = raw_text.strip()
    lowered = stripped.lower().rstrip("?").strip()
    if lowered in _NOT_ACTUALLY_A_QUESTION:
        return False
    if stripped.endswith("?"):
        return True
    return any(lowered.startswith(starter) for starter in _QUESTION_STARTERS)


def _has_unaddressed_caveat(raw_text: str) -> bool:
    return bool(re.search(r"\bbut\b", raw_text, re.IGNORECASE))

_CLEAN_YES_MAX_WORDS = 6


def _is_clean_affirmative(user_message: str, clean: str) -> bool:
    if len(clean.split()) > _CLEAN_YES_MAX_WORDS:
        return False
    if _looks_like_a_question(user_message):
        return False
    if _has_unaddressed_caveat(user_message):
        return False
    return _contains_phrase(clean, SPECIALIST_YES_PHRASES) or bool(
        _fuzzy_match_short_phrase(clean, SPECIALIST_YES_PHRASES)
    )


def _is_emphatic_repeated_rejection(user_message: str) -> bool:
    segments = [s.strip() for s in re.split(r"[.!]+", user_message) if s.strip()]
    if len(segments) < 2:
        return False
    return all(
        _contains_phrase(_normalize(seg), REJECTION_PHRASES)
        or bool(_fuzzy_match_short_phrase(_normalize(seg), REJECTION_PHRASES))
        for seg in segments
    )
def _is_clean_rejection(user_message: str, clean: str) -> bool:
    if len(clean.split()) > _CLEAN_YES_MAX_WORDS:
        return False
    if _looks_like_a_question(user_message):
        return False
    if _has_unaddressed_caveat(user_message):
        return False
    return _contains_phrase(clean, REJECTION_PHRASES) or bool(
        _fuzzy_match_short_phrase(clean, REJECTION_PHRASES)
    )

_SEND_DETAILS_RE = re.compile(
    r"\b(?:send|share|text|whatsapp|email|forward|message)\b.{0,30}"
    r"\b(?:details?|info(?:rmation)?|profile|brochure|documents?|"
    r"catalog(?:ue)?|materials?)\b",
    re.IGNORECASE,
)


def _asks_to_send_details(raw_text: str) -> bool:
    return bool(_SEND_DETAILS_RE.search(raw_text))

_NAME_CORRECTION_RE = re.compile(
    r"\b(?:my name is|it'?s|i'?m)\s+([a-zA-Z](?:[\s-]?[a-zA-Z])*)",
    re.IGNORECASE,
)
_ORG_CORRECTION_RE = re.compile(
    r"\b(?:i'?m calling from|we'?re calling from|calling from|"
    r"my company is|my organization is|it'?s)\s+([A-Za-z0-9&.\- ]+?)"
    r"(?:[.,!?]|$)",
    re.IGNORECASE,
)


def _reassemble_spelled_name(candidate: str) -> str:
    parts = re.split(r"[\s-]+", candidate.strip())
    if len(parts) >= 3 and all(len(p) == 1 for p in parts):
        return "".join(parts).capitalize()
    return candidate


_SPELL_SEPARATOR_RE = re.compile(r"[\s,-]+")


def _extract_spelled_name(text: str) -> str:
    cleaned = _NAME_PREFIX_RE.sub("", text.strip(), count=1)
    tokens = _SPELL_SEPARATOR_RE.split(cleaned.strip())
    letters = [t for t in tokens if len(t) == 1 and t.isalpha()]
    if len(letters) < 2:
        return ""
    return "".join(letters).capitalize()


def _detect_name_correction(text: str):
    match = _NAME_CORRECTION_RE.search(text)
    if not match:
        return None
    return _reassemble_spelled_name(match.group(1).strip())


def _detect_org_correction(text: str):
    match = _ORG_CORRECTION_RE.search(text)
    return match.group(1).strip() if match else None

_NAME_NEGATION_RE = re.compile(
    r"\bmy name (?:is ?not|isnt)\b|\bthats not my name\b|"
    r"\bthat is not my name\b|\bnot my name\b|\bwrong name\b"
)


def _is_name_negation_without_replacement(clean_text: str) -> bool:
    return bool(_NAME_NEGATION_RE.search(clean_text))

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


def _is_busy_reply(clean: str) -> bool:
    return bool(
        _contains_phrase(clean, BUSY_PHRASES)
        or _fuzzy_match_short_phrase(clean, BUSY_PHRASES)
    )

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

_PRINTER_BRAND_ALIASES = {
    "zebra": "Zebra",
    "evolis": "Evolis",
    "hid fargo": "HID Fargo",
    "fargo": "HID Fargo",
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

        self.waiting_for_callback_confirmation = False
        self._specialist_offer_asked_for_time = False
        self.last_prompt_eval_seconds = None

        self.llm = llm if llm is not None else LLM()
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase()
        self.call_started = False
        self.waiting_for_opening_pitch = False
        self.waiting_for_minute_response = False
        self.last_reply = None
        self.call_finished = False
        self.waiting_for_name = False
        self.waiting_for_name_spelling = False
        self.waiting_for_organization = False
        self.waiting_for_callback_time = False
        self.waiting_for_final_correction = False
        self.waiting_for_product_or_callback_choice = False
        self.opening_choice_asked = False
        self.controller = ConversationController()

        self.customer_name = None
        self.customer_organization = None
        self.customer_callback_time = None
        self._time_retry_count = 0
        self._name_retry_count = 0
        self._org_retry_count = 0
        self._opening_silence_retry_count = 0
        self._pending_collection_question = None
        self._pending_callback_time = None
        self.waiting_for_name_confirmation = False
        self._name_confirmation_retry_count = 0
        self._last_ack_filler_opener = None
        self._pending_brand = None
        self.brands_offered_ribbons_for = set()
        self.specialist_offer_count = 0
        self.exchange_count = 0
        self.call_id = uuid.uuid4().hex
        self.transcript = []
        self._transcript_path = TRANSCRIPT_DIR / f"{self.call_id}.jsonl"
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
        self.last_reply = text
        return text

    def start_call_opening(self):
        self.call_started = True
        self.waiting_for_opening_pitch = True
        return self._reply(OPENING_GREETING)

    def respond_low_confidence(self):
        return self._reply("Sorry, could you say that again?")

    def save_summary(self, extra=None):
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

    def _try_fast_path(self, user_message, clean, on_sentence=None):

        # Opening Message
        if not self.call_started:
            self.call_started = True
            self.waiting_for_opening_pitch = True
            return self._reply(OPENING_GREETING)

        if _is_emphatic_repeated_rejection(user_message):
            self.call_finished = True
            self.controller.reject()
            return self._reply("No problem, thank you for your time, have a great day!")

        if self.waiting_for_opening_pitch:
            if user_message == NO_RESPONSE_SENTINEL:
                self._opening_silence_retry_count += 1
                if self._opening_silence_retry_count >= 2:
                    self.waiting_for_opening_pitch = False
                    self.call_finished = True
                    self.controller.reject()
                    return self._reply("No problem, thank you for your time, have a great day!")
                return self._reply("I cannot hear you, are you still there?")

            self.waiting_for_opening_pitch = False
            if _is_clean_rejection(user_message, clean):
                self.call_finished = True
                self.controller.reject()
                return self._reply("No problem, thank you for your time, have a great day!")
            if _is_busy_reply(clean):
                self.waiting_for_callback_confirmation = True
                self._specialist_offer_asked_for_time = True
                self.controller.busy()
                return self._reply("No problem. Would another time be better?")
            self.waiting_for_minute_response = True
            return self._reply(OPENING_PITCH)

        # only the turn right after "do you have a quick minute?" should trigger this
        if self.waiting_for_minute_response:
            self.waiting_for_minute_response = False
            if not self.opening_choice_asked and _is_clean_affirmative(user_message, clean):
                self.opening_choice_asked = True
                self.waiting_for_product_or_callback_choice = True
                return self._reply(
                    "Great. Would you like to know about a specific product "
                    "or service, or would you prefer I connect you with a "
                    "specialist for a callback?"
                )

        if self.waiting_for_product_or_callback_choice:
            self.waiting_for_product_or_callback_choice = False
            if _is_clean_affirmative(user_message, clean):
                wants_callback = True
            elif _is_clean_rejection(user_message, clean):
                return self._reply(
                    "No worries, is there anything I can help with, or "
                    "would you like me to let you go for now?"
                )
            elif _looks_like_a_question(user_message):
                wants_callback = False
            else:
                wants_callback = _llm_classify_product_or_callback(self.llm, user_message)

            if wants_callback:
                self.waiting_for_name = True
                return self._reply("Great. May I get your name?")

        if self.waiting_for_final_correction:
            self.waiting_for_final_correction = False
            new_time = _llm_extract_field(
                self.llm,
                "callback time (day and/or time of day)",
                "What time would you prefer for the callback?",
                user_message,
            )
            if not new_time:
                new_time = _extract_time_phrase(user_message)

            if (
                new_time
                and _looks_like_time(new_time)
                and _extracted_value_is_grounded(new_time, user_message)
                and new_time.strip().lower() != (self.customer_callback_time or "").strip().lower()
            ):
                self.customer_callback_time = new_time
                self.controller.set_callback_time(new_time)
                self.call_finished = True
                return self._reply(
                    f"Got it, we'll call you at {new_time} instead. "
                    f"Thanks again, have a great day!"
                )

            self.call_finished = True
            return self._reply("Understood, thank you again, have a great day!")
        if self.last_reply and _requests_repeat(clean):
            original = self.last_reply
            result = self._reply(f"Sorry, let me repeat that — {original}")

            self.last_reply = original
            return result

        if self.customer_name and (
            "my name is" in clean
            or "actually" in clean
            or "correction" in clean
            or re.search(r"\bnot\b", clean)
        ):
            corrected = _detect_name_correction(user_message)
            if (
                corrected
                and len(corrected) >= 2
                and _looks_like_name(corrected)
                and corrected.lower() != self.customer_name.lower()
            ):
                self.customer_name = corrected.title()
                print(f"[CHAT] Name corrected to: {self.customer_name}")
                self.waiting_for_name_confirmation = False
                self.controller.set_contact_details(
                    self.customer_name, self.customer_organization
                )
                return self._reply(
                    f"Got it, I've updated your name to {self.customer_name}."
                )
            if _is_name_negation_without_replacement(clean):
                self.customer_name = None
                self.waiting_for_name = True
                self.waiting_for_name_confirmation = False
                return self._reply("Sorry, what's your actual name?")

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
                    f"Got it, I've updated your organization to "
                    f"{self.customer_organization}."
                )
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
                    f"Got it, I've updated the callback time to {new_time}."
                )
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

        if self.waiting_for_name_confirmation:
            if _is_location_question(user_message):
                return self._reply(
                    f"{SPARK_LOCATION_ANSWER} Now, is {self.customer_name} correct?"
                )

            if (
                self.customer_name
                and self.customer_name.lower() in clean
                and not _contains_phrase(clean, REJECTION_PHRASES)
                and not _fuzzy_match_short_phrase(clean, REJECTION_PHRASES)
            ):
                self.waiting_for_name_confirmation = False
                self.waiting_for_organization = True
                return self._reply("And which organization are you calling from?")

            if _is_clean_affirmative(user_message, clean):
                self.waiting_for_name_confirmation = False
                self.waiting_for_organization = True
                return self._reply("And which organization are you calling from?")

            corrected = _detect_name_correction(user_message)
            if corrected and len(corrected) >= 2 and _looks_like_name(corrected):
                self.waiting_for_name_confirmation = False
                self.customer_name = corrected.title()
                self.waiting_for_organization = True
                return self._reply(
                    f"Got it, {self.customer_name}. And which "
                    f"organization are you calling from?"
                )

            if _is_clean_rejection(user_message, clean):
                self.waiting_for_name_confirmation = False
                self.customer_name = None
                self.waiting_for_name = True
                self.waiting_for_name_spelling = False
                self._name_retry_count = 0
                return self._reply("Sorry about that, what's your name?")

            self._name_confirmation_retry_count += 1
            if self._name_confirmation_retry_count >= 2:
                self.waiting_for_name_confirmation = False
                self.waiting_for_organization = True
                return self._reply(
                    "Let's continue, and which organization are you "
                    "calling from?"
                )
            return self._reply(
                f"Sorry, just to confirm — is {self.customer_name} correct?"
            )

        # Rejection
        if clean in REJECTION_PHRASES:
            self.controller.reject()
            self.call_finished = True
            return self._reply("No problem at all. Thanks for your time. Have a great day.")

        # Busy
        if _is_busy_reply(clean):
            self.waiting_for_callback_confirmation = True
            self._specialist_offer_asked_for_time = True
            self.controller.busy()
            return self._reply("No problem. Would another time be better?")

        # Callback confirmation response
        if self.waiting_for_callback_confirmation:

            if _is_clean_affirmative(user_message, clean):
                self.waiting_for_callback_confirmation = False
                self.waiting_for_name = True
                self.controller.start_scheduling()
                return self._reply("Great. May I get your name?")

            if _is_clean_rejection(user_message, clean):
                self.waiting_for_callback_confirmation = False
                self.controller.reject()
                self.call_finished = True
                return self._reply("No problem, thank you for your time!")

            if self._specialist_offer_asked_for_time and _looks_like_time(user_message):
                self.waiting_for_callback_confirmation = False
                self._pending_callback_time = _extract_time_phrase(user_message)
                self.waiting_for_name = True
                self.controller.start_scheduling()
                return self._reply("Great. May I get your name?")

            self.waiting_for_callback_confirmation = False
            if len(clean.split()) >= 3:
                self.specialist_offer_count = 0

        # Name and organization — two separate deterministic questions.
        if self.waiting_for_name:
            if _is_location_question(user_message):
                return self._reply(
                    f"{SPARK_LOCATION_ANSWER} Now, could I get your name?"
                )
            if _looks_like_a_question(user_message):

                self._pending_collection_question = (user_message, "May I get your name?")
                return None

            name_candidate = _extract_name_answer(user_message)
            if not name_candidate:
                name_candidate = _llm_extract_field(
                    self.llm, "customer's name", "May I get your name?", user_message
                )

            if not _looks_like_name(name_candidate) or not _extracted_value_is_grounded(name_candidate, user_message):
                self._name_retry_count += 1
                if self._name_retry_count >= 2:
                    self.waiting_for_name = False
                    self.waiting_for_name_spelling = True
                    return self._reply(
                        "Sorry, I'm still not catching that — could "
                        "you spell your name out for me, letter by "
                        "letter?"
                    )
                return self._reply(
                    "Sorry, I didn't catch your name — could you say "
                    "that again?"
                )

            self.waiting_for_name = False
            self.customer_name = name_candidate.title()
            self.waiting_for_name_confirmation = True
            self._name_confirmation_retry_count = 0
            return self._reply(f"Okay — {self.customer_name}, is that right?")

        if self.waiting_for_name_spelling:
            if _is_location_question(user_message):
                return self._reply(
                    f"{SPARK_LOCATION_ANSWER} Could you spell your "
                    f"name out for me, letter by letter?"
                )
            if _looks_like_a_question(user_message):
                self._pending_collection_question = (
                    user_message,
                    "Could you spell your name out for me, letter by letter?",
                )
                return None

            spelled_name = _extract_spelled_name(user_message)
            if not spelled_name:
                self.waiting_for_name_spelling = False
                self.waiting_for_organization = True
                return self._reply(
                    "No worries — let's move on. Which organization "
                    "are you calling from?"
                )

            self.waiting_for_name_spelling = False
            self.customer_name = spelled_name
            self.waiting_for_name_confirmation = True
            self._name_confirmation_retry_count = 0
            return self._reply(f"Okay — {self.customer_name}, is that right?")

        if self.waiting_for_organization:
            if _is_location_question(user_message):
                return self._reply(
                    f"{SPARK_LOCATION_ANSWER} Now, which organization are you calling from?"
                )
            if _looks_like_a_question(user_message):
                self._pending_collection_question = (user_message, "Which organization are you calling from?")
                return None

            org_candidate = _extract_org_answer(user_message)
            if not org_candidate:
                org_candidate = _llm_extract_field(
                    self.llm,
                    "organization or company name",
                    "Which organization are you calling from?",
                    user_message,
                )

            if not _looks_like_org(org_candidate) or not _extracted_value_is_grounded(org_candidate, user_message):
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
                callback_time = self._pending_callback_time
                self._pending_callback_time = None
                self.customer_callback_time = callback_time
                self.controller.set_callback_time(callback_time)
                self.controller.close()
                self.waiting_for_final_correction = True
                name_part = f"{self.customer_name}, " if self.customer_name else ""
                return self._reply(
                    f"Perfect, {name_part}our specialist will contact you "
                    f"{callback_time} on this same number. Thank you "
                    f"for your time, have a great day!"
                )

            self.waiting_for_callback_time = True
            return self._reply("Okay. What time would you prefer for the callback?")

        # Customer directly requests a callback, skipping confirmation.
        if clean in SCHEDULING_PHRASES:
            self.waiting_for_name = True
            self.controller.start_scheduling()
            return self._reply("Certainly. May I get your name?")

        # Callback time.
        if self.waiting_for_callback_time:
            if _is_location_question(user_message):
                return self._reply(
                    f"{SPARK_LOCATION_ANSWER} Now, what time would you prefer for the callback?"
                )
            if _looks_like_a_question(user_message):
                self._pending_collection_question = (user_message, "What time would you prefer for the callback?")
                return None

            remaining_text, extracted_name = _extract_and_strip_name(user_message)
            if extracted_name and not self.customer_name:
                self.customer_name = extracted_name.title()
                self.controller.set_contact_details(
                    self.customer_name, self.customer_organization
                )

            time_candidate = _extract_time_phrase(remaining_text)
            if not time_candidate:
                time_candidate = _llm_extract_field(
                    self.llm,
                    "callback time (day and/or time of day)",
                    "What time would you prefer for the callback?",
                    remaining_text,
                )

            if not _looks_like_time(time_candidate) or not _extracted_value_is_grounded(time_candidate, remaining_text):
                self._time_retry_count += 1
                if self._time_retry_count >= 2:
                    self.waiting_for_callback_time = False
                    self.controller.set_callback_time(None)
                    self.controller.close()
                    self.call_finished = True
                    return self._reply(
                        "No problem — our specialist will reach out on "
                        "this same number to find a time that works. "
                        "Thank you for your time, have a great day!"
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
            self.waiting_for_final_correction = True

            name_part = f"{self.customer_name}, " if self.customer_name else ""
            return self._reply(
                f"Perfect, {name_part}our specialist will contact you "
                f"{callback_time} on this same number. Thank you for "
                f"your time, have a great day!"
            )
        if clean in GREETING_PHRASES:
            return self._reply("Hi, yes — I'm here, please go ahead.")
        if clean in GRATITUDE_PHRASES:
            return self._reply(
                "You're welcome — is there anything else I can help "
                "with regarding our ID card solutions?"
            )

        return None

    def _build_llm_messages(self, user_message, clean):
        self.exchange_count += 1

        if clean in GENERIC_CONTINUATIONS:
            knowledge_context = ""
        else:
            start = time.perf_counter()
            knowledge_results = self.knowledge.search(user_message, n_results=2)
            if DEBUG_TIMING:
                print(f"[TIMING] Knowledge search: {time.perf_counter() - start:.2f}s")
            knowledge_context = "\n\n".join(knowledge_results)

        if self._pending_collection_question is not None:
            _, pending_field_question = self._pending_collection_question
            self._pending_collection_question = None
            stage_note = (
                "the customer asked a question while you were in the "
                "middle of collecting information from them. Answer "
                "their question directly and briefly using the "
                "knowledge above if relevant — never invent facts not "
                "in the knowledge base. Then, in this same reply, ask "
                f"exactly this to continue: \"{pending_field_question}\""
            )
        elif self.exchange_count >= 2:
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
        combined_content = f"{user_message}\n\n[{knowledge_block}]"
        self.messages.append({"role": "user", "content": combined_content})

        llm_messages = list(self.messages)

        if DEBUG_TIMING:
            prompt_chars = sum(len(m["content"]) for m in llm_messages)
            print(f"[TIMING] Prompt size: {prompt_chars} characters")

        if DEBUG_PROMPT_DUMP:
            print(f"\n===== FULL PROMPT DUMP (turn, {len(llm_messages)} messages) =====")
            for i, msg in enumerate(llm_messages):
                print(f"--- MESSAGE {i} [{msg['role']}] ---")
                print(repr(msg["content"]))
            print("===== END PROMPT DUMP =====\n")

        return llm_messages

    def _maybe_add_ribbon_fallback(self, reply, on_sentence=None):
        brand = self._pending_brand
        self._pending_brand = None

        if not brand or brand in self.brands_offered_ribbons_for:
            return reply

        self.brands_offered_ribbons_for.add(brand)

        if "wholesale" in reply.lower() or "ribbon" in reply.lower():
            return reply

        fallback_sentence = (
            f"We also offer wholesale rates on {brand} "
            f"printer ribbons."
        )

        if on_sentence:
            on_sentence(fallback_sentence)

        reply = reply.rstrip()
        if reply and reply[-1] not in ".!?":
            reply += "."
        return f"{reply} {fallback_sentence}"

    def _finalize_llm_reply(self, reply, on_sentence=None):
        if not reply.strip():
            reply = "Sorry, I didn't quite catch that — could you say that again?"
            if on_sentence:
                on_sentence(reply)

        if _offers_specialist(reply):
            self.specialist_offer_count += 1
            if self.specialist_offer_count >= 3:
                self.controller.reject()
                self.call_finished = True
                reply = (
                    "No problem at all — I'll leave you to it. Thanks "
                    "for your time, have a great day!"
                )
                if on_sentence:
                    on_sentence(reply)
            else:
                self.waiting_for_callback_confirmation = True
                self._specialist_offer_asked_for_time = bool(
                    re.search(r"what time|when would|when works|what day", reply, re.IGNORECASE)
                )
                self.controller.specialist_requested()
        elif _sounds_like_closing(reply):
            self.controller.close()
            self.call_finished = True

        self.messages.append({"role": "assistant", "content": reply})

        if len(self.messages) > 15:
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
        user_message = user_message.strip()
        total_start = time.perf_counter()

        if not user_message:
            reply = "Sorry, I didn't catch that."
            on_sentence(reply)
            return reply

        self._log("user", user_message)
        clean = _normalize(user_message)

        fast_reply = self._try_fast_path(user_message, clean, on_sentence=on_sentence)
        if fast_reply is not None:
            on_sentence(fast_reply)
            return fast_reply

        llm_messages = self._build_llm_messages(user_message, clean)

        start = time.perf_counter()
        full_sentences = []
        first_sentence_at = None
        first_sentence_ready = threading.Event()

        def _stall_filler():
            if not first_sentence_ready.wait(timeout=_STALL_THRESHOLD_SECONDS):
                available = [
                    p for p in _STALL_PHRASES
                    if p.split()[0].lower() != self._last_ack_filler_opener
                ]
                phrase = random.choice(available)
                self._last_ack_filler_opener = phrase.split()[0].lower()
                self._log("assistant", phrase)
                on_sentence(phrase)

        stall_thread = threading.Thread(target=_stall_filler, daemon=True)
        stall_thread.start()

        _llm_result = {}
        for sentence in _stream_to_sentences(self.llm.generate_stream(llm_messages, result_holder=_llm_result)):
            if first_sentence_at is None:
                first_sentence_at = time.perf_counter()
                if DEBUG_TIMING:
                    print(f"[TIMING] Time to first sentence: {first_sentence_at - start:.2f}s")
                first_sentence_ready.set()
                stall_thread.join()
            full_sentences.append(sentence)
            on_sentence(sentence)

        if DEBUG_TIMING:
            print(f"[TIMING] LLM generation (full): {time.perf_counter() - start:.2f}s")
        self.last_prompt_eval_seconds = _llm_result.get("prompt_eval_seconds")

        reply = " ".join(full_sentences)
        reply = self._maybe_add_ribbon_fallback(reply, on_sentence=on_sentence)
        result = self._finalize_llm_reply(reply, on_sentence=on_sentence)

        if DEBUG_TIMING:
            print(f"[TIMING] Total response: {time.perf_counter() - total_start:.2f}s")

        return result
