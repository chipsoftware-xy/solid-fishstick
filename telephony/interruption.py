"""
Interruption detection for audio arriving while Spark is actively
speaking — built to directly implement a specific, correct behavioral
spec: a real sales agent keeps listening while talking, ignores pure
backchannel ("mm-hmm", "yeah"), but if the customer says something
substantive, finishes their current sentence and then addresses it,
rather than either ignoring it entirely or cutting itself off mid-word.

Deliberately NOT built on raw-audio-signal echo cancellation
(VoiceClean) — that approach was attempted three separate times
tonight (no echo cancellation, a blind upfront reference feed, a
real-time-paced incremental reference feed) and failed the same way
each time on real calls. This takes a fundamentally different approach:
since we know, with certainty, the exact text Spark is currently
speaking (we generated it ourselves), audio picked up while Spark talks
gets transcribed and compared against Spark's own known words —
matching content is treated as echo and ignored, genuinely different
content is treated as a real interruption. This trades a couple of
seconds of latency (an STT round-trip on the suspected-interruption
audio) for not needing to solve real-time signal-level echo
cancellation at all — a trade the user explicitly said is acceptable.
"""
import re

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", text.strip().lower())


# Pure backchannel — a real listener makes these sounds constantly while
# someone else is talking, and none of them call for a response. Kept
# separate from chat_session.py's SPECIALIST_YES_PHRASES/REJECTION_PHRASES
# deliberately: those are meaningful ANSWERS to a specific yes/no
# question asked in a specific state, evaluated after Spark has finished
# speaking. This set is evaluated purely on its own, independent of any
# question — the customer going "mm-hmm" is never itself an answer to
# anything, no matter what Spark just asked.
FILLER_PHRASES = {
    "mm", "mhm", "mmhmm", "mm hmm", "uh huh", "uhhuh", "huh",
    "yeah", "yep", "yup", "ok", "okay", "alright", "right",
    "i see", "got it", "sure", "cool", "nice", "oh", "ah", "hmm",
}


def is_filler(text: str) -> bool:
    """
    True only if the ENTIRE utterance is filler — every word-group in
    it matches a known backchannel phrase. A longer utterance that
    happens to start with "okay" ("Okay, but can you deliver to
    Mumbai?") is NOT filler; only bare backchannel on its own counts.
    Checked against the full normalized string first (covers the
    common single-phrase case directly), then falls back to requiring
    every individual word to itself be a filler word, so "yeah yeah
    okay" (multiple filler words strung together, still nothing
    substantive) is correctly caught too.
    """
    normalized = _normalize(text)
    if not normalized:
        return True  # Genuinely empty audio, not something to act on.
    if normalized in FILLER_PHRASES:
        return True
    words = normalized.split()
    return all(w in FILLER_PHRASES for w in words)


def is_likely_echo(candidate_text: str, sparks_current_text: str, overlap_threshold: float = 0.6) -> bool:
    """
    Compares a suspected-interruption transcript against the text
    Spark is currently, actually speaking. Real acoustic echo of
    Spark's own voice, even imperfectly transcribed by STT (phone-line
    degradation, partial capture), should still substantially overlap
    with Spark's real words — whereas genuine, different customer
    speech generally won't share much vocabulary with whatever Spark
    happens to be saying at that exact moment.

    Word-overlap ratio, not exact/substring matching — a garbled
    partial echo transcript won't be a clean substring of the source,
    but should still share most of its actual words. overlap_threshold
    of 0.6 means: if 60%+ of the candidate's words appear in what
    Spark is currently saying, treat it as echo. This is a heuristic
    the user explicitly accepted the tradeoffs of (a couple seconds of
    added latency in exchange for not fighting raw-audio echo
    cancellation) — not a guarantee, and worth tuning after real calls
    if it's too aggressive or too lax in practice.
    """
    candidate_words = set(_normalize(candidate_text).split())
    if not candidate_words:
        return True  # Nothing real was said — treat as noise/echo, not an interruption.

    spark_words = set(_normalize(sparks_current_text).split())
    if not spark_words:
        return False  # Spark has no known current text to compare against — can't call it echo.

    overlap = candidate_words & spark_words
    return (len(overlap) / len(candidate_words)) >= overlap_threshold


def should_interrupt(candidate_text: str, sparks_current_text: str) -> bool:
    """
    The actual decision point: only interrupt Spark if the candidate
    text is BOTH genuinely different from Spark's own current speech
    (not echo) AND substantive (not pure backchannel). Either check
    failing means "keep talking" — a real sales agent doesn't stop
    mid-sentence for "mm-hmm", and shouldn't stop because it heard its
    own voice bounce back either.
    """
    if is_filler(candidate_text):
        return False
    if is_likely_echo(candidate_text, sparks_current_text):
        return False
    return True