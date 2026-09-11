"""
Two-way telephony pipeline test — milestone 3, the real thing.

Wires together everything proven so far (turn detection, dual-tier
STT) with Spark's actual brain (ChatSession) and voice (Piper TTS),
over a real phone call. This is the first point where Spark can
actually hold a real conversation over the phone, not just have its
STT quality inspected in isolation.

Architecture note: chat_session.send_streaming() is synchronous/
blocking (built for local playback, where blocking is fine) — it runs
in a background thread here via asyncio.to_thread(), with each
synthesized sentence's audio handed off through a thread-safe queue to
the async side, which sends it over the WebSocket. This keeps the
event loop from freezing for the whole duration of a reply.

Run from the PROJECT ROOT:
    python -m telephony.two_way_test

Setup (same as before, nothing new):
    pip install websockets webrtcvad-wheels --break-system-packages
"""
import asyncio
import base64
import gc
import json
import queue
import time
import uuid
from pathlib import Path

import psutil

import websockets
import numpy as np

from telephony.stream_vad import CallTurnDetector, _compute_rms
from telephony.audio_utils import resample_pcm16, pcm16_to_whisper_input
from telephony.dual_tier_stt import DualTierSTT
from app.llm.chat_session import GRATITUDE_PHRASES
from telephony.dialer import place_call, get_exotel_settings, ExotelConfigError, normalize_phone
from app.tts.piper import TTS
from app.llm.chat_session import ChatSession
from app.llm.ollama_client import LLM
from app.knowledge_base import KnowledgeBase
from app.call_finalizer import finalize_call
from app.call_storage import mark_done_by_phone, mark_failed_by_phone, get_next_pending, set_queue_status, save_call, init_db

import app.llm.chat_session as chat_session
import app.call_storage as call_storage

VAD_SETTINGS = {
    "vad_aggressiveness": 2,
    "vad_frame_ms": 30,
    "vad_preroll_ms": 300,
    "vad_silence_ms": 800,
    "vad_min_speech_ms": 300,
    "vad_max_utterance_ms": 20000,
    # Raised from 150 based on real diagnostic data: every hallucinated
    # "Thank you" (breathing/faint noise near the mic, per direct
    # testing) measured RMS 184-597, while every confirmed genuine
    # utterance measured 948+. 700 sits comfortably between the two
    # observed populations with margin on both sides.
    "vad_min_rms": 700,
}

WHISPER_SAMPLE_RATE = 16000
EXOTEL_SAMPLE_RATE = 8000

# Confirmed via direct measurement: this process's memory footprint
# grows call over call (not a reference leak - gc.collect() confirmed
# it can't reclaim it, since CPython's allocator just doesn't hand
# freed memory back to the OS), and that growth measurably correlates
# with slower first-turn LLM latency on subsequent calls. Restarting
# the whole process periodically is the only reliable fix - see the
# full explanation at the restart-trigger site below.
#
# Primary trigger is REACTIVE (turn 1's actual prompt-eval time), not
# a fixed call count - confirmed directly that the slowdown can hit as
# early as the SECOND call (1061MB baseline -> 1.88s vs. 1345MB
# baseline -> 9.39s, back to back). A fixed count high enough to be
# safe would let most calls in a batch eat the slow-turn-1 penalty
# before ever restarting. MAX_CALLS_BEFORE_RESTART is kept as a safety
# net only, in case the threshold happens not to fire for a while.
FIRST_TURN_EVAL_RESTART_THRESHOLD_SECONDS = 5.0
MAX_CALLS_BEFORE_RESTART = 20
_completed_call_count = 0
_shutdown_event = None  # created inside main(), since it needs a running event loop
# Within Exotel's documented 3200-100000 byte range, a multiple of 320.
# 3200 bytes at 8kHz/16-bit mono = 200ms of audio per outgoing chunk.
OUTGOING_CHUNK_BYTES = 3200

# Simple file-based signals for cross-process coordination with Flask
# — no sockets/IPC needed for something this infrequent. BATCH_ACTIVE
# exists exactly while an auto-dial chain is running (distinguishes a
# batch call from a one-off manual "Call" click, which should NOT
# auto-continue). BATCH_STOP_SIGNAL is written by Flask's Stop button;
# checked before and after the pause, so Stop takes effect promptly
# either way. ACTIVE_CALL_ID lets Flask's live-transcript endpoint know
# which transcripts/{id}.jsonl file to tail, since that ChatSession
# object lives in THIS process, not Flask's.
BATCH_ACTIVE_PATH = Path("data/batch_active")
BATCH_STOP_SIGNAL_PATH = Path("data/batch_stop_signal")
ACTIVE_CALL_ID_PATH = Path("data/active_call_id.txt")
PAUSE_BETWEEN_CALLS_SECONDS = 90  # 1.5 minutes

# Real testing data: confirmed hallucinations/deliberate-gibberish
# transcriptions measured avg_logprob -1.25 to -3.27; confirmed genuine
# (even imperfect) transcriptions measured -0.29 to -0.87. -1.1 sits
# cleanly between the two observed populations.
CONFIDENCE_REJECT_THRESHOLD = -1.1

# Loaded once, at import time — shared across every call, same
# principle as VoiceSession locally. Only the ChatSession itself is
# created fresh per call.
print("Loading shared resources (this happens once)...")
stt = DualTierSTT(fast_model="base", accurate_model="small")
tts = TTS()
llm = LLM()
knowledge = KnowledgeBase()

# Warms up both Whisper tiers with a throwaway call each, same
# principle as the existing Ollama warmup in ollama_client.py.
# Confirmed real cost: a customer's actual first turn measured
# stt_time=17.76s and returned an empty transcript — the model object
# being constructed doesn't mean the first real inference call is
# fast; there's a separate, one-time cost (kernel compilation, first
# touch of memory-mapped weights) that only shows up on first genuine
# use. Paying that here, before any real call can arrive, instead of
# a real customer sitting through it as dead air.
print("Warming up STT models...")
_warmup_audio = np.zeros(16000, dtype=np.float32)  # 1s of silence at 16kHz
stt.transcribe_array(_warmup_audio, duration_s=1.0)   # warms the "small" tier
stt.transcribe_array(_warmup_audio, duration_s=3.0)   # warms the "base" tier
print("STT models warmed up.")

# Confirmed via direct recorded-audio review of a real call: the
# silent window between the customer finishing speaking and Spark's
# first response is entirely unmasked — the existing stall filler only
# ever covers LLM generation time, not the STT window sitting in front
# of it (1.3-6+ seconds observed on real phone audio), which is what
# actually produced the "not listening"/"writing it down" impression.
# Pre-synthesized once here (not per-turn) so playing it costs nothing
# but the send itself - real people backchannel ("mm-hmm", "okay")
# constantly while listening, so this mirrors real conversation rather
# than just masking a technical delay.
print("Pre-synthesizing acknowledgment sound...")
_ack_pcm_bytes, _ack_native_rate = tts.synthesize_to_pcm("Okay.")
ACKNOWLEDGMENT_PCM_8K = resample_pcm16(_ack_pcm_bytes, _ack_native_rate, EXOTEL_SAMPLE_RATE).tobytes()
ACKNOWLEDGMENT_SECONDS = len(ACKNOWLEDGMENT_PCM_8K) / 2 / EXOTEL_SAMPLE_RATE
print(f"Acknowledgment sound ready ({ACKNOWLEDGMENT_SECONDS:.2f}s).")

print("Shared resources ready.")


SEND_TIMEOUT_SECONDS = 30


async def send_pcm_as_media(ws, stream_sid, pcm_bytes):
    """Splits PCM audio into Exotel-sized chunks and sends each as its
    own media event. Wrapped in a short timeout, not because a genuine
    conversation has any length limit, but because sending bytes over
    an already-open connection should never legitimately take this
    long — if it does, the connection is effectively dead even though
    nothing has raised ConnectionClosed yet, which is a real observed
    failure mode (a 6+ minute hang after a customer hung up), not a
    theoretical one. This is a targeted timeout on the specific
    operation that hung, not a limit on overall call duration."""
    async def _send():
        for i in range(0, len(pcm_bytes), OUTGOING_CHUNK_BYTES):
            chunk = pcm_bytes[i:i + OUTGOING_CHUNK_BYTES]
            payload = base64.b64encode(chunk).decode()
            await ws.send(json.dumps({
                "event": "media",
                "stream_sid": stream_sid,
                "media": {"payload": payload},
            }))

    await asyncio.wait_for(_send(), timeout=SEND_TIMEOUT_SECONDS)


async def run_turn(ws, chat, stream_sid, user_message):
    """
    Runs one full Spark turn: chat_session.send_streaming() in a
    background thread (it's synchronous), streaming each sentence's
    synthesized audio back over the WebSocket as it becomes ready
    rather than waiting for the entire reply to finish. Returns the
    total duration (seconds) of audio actually sent, so the caller can
    tell "bytes fully sent" apart from "customer has actually heard
    it" — closing the connection right after this returns would still
    cut audio off mid-playback, since sending completes almost
    instantly while playback takes the audio's real duration.
    """
    audio_queue = queue.Queue()
    SENTINEL = object()
    total_audio_seconds = 0.0

    def on_sentence(sentence_text):
        nonlocal total_audio_seconds
        print(f"[SPARK] {sentence_text}")
        pcm_bytes, native_rate = tts.synthesize_to_pcm(sentence_text)
        pcm_8k = resample_pcm16(pcm_bytes, native_rate, EXOTEL_SAMPLE_RATE).tobytes()
        # PCM16 = 2 bytes per sample.
        total_audio_seconds += len(pcm_8k) / 2 / EXOTEL_SAMPLE_RATE
        audio_queue.put(pcm_8k)

    def run_chat():
        try:
            chat.send_streaming(user_message, on_sentence)
        finally:
            audio_queue.put(SENTINEL)

    thread_task = asyncio.create_task(asyncio.to_thread(run_chat))

    while True:
        item = await asyncio.to_thread(audio_queue.get)
        if item is SENTINEL:
            break
        await send_pcm_as_media(ws, stream_sid, item)

    await thread_task
    return total_audio_seconds


async def _continue_batch():
    """
    Called right after a call that was part of an active batch
    finishes. Pauses, then dials the next pending number — or ends the
    batch cleanly if the queue's empty or Stop was clicked. This is
    what keeps the auto-dial chain going without Flask needing to
    signal anything after the first call — this server already knows
    the exact instant each call ends, so it's the natural place for
    the "what happens next" decision to live.
    """
    if BATCH_STOP_SIGNAL_PATH.exists():
        print("[BATCH] Stop requested — ending batch.")
        BATCH_STOP_SIGNAL_PATH.unlink()
        BATCH_ACTIVE_PATH.unlink(missing_ok=True)
        return

    print(f"[BATCH] Pausing {PAUSE_BETWEEN_CALLS_SECONDS}s before next call...")
    # asyncio.sleep, NOT time.sleep — a blocking sleep here would
    # freeze the entire event loop for the whole pause, preventing any
    # other connection from being handled during that time.
    await asyncio.sleep(PAUSE_BETWEEN_CALLS_SECONDS)

    # Re-check — Stop may have been clicked WHILE waiting.
    if BATCH_STOP_SIGNAL_PATH.exists():
        print("[BATCH] Stop requested during pause — ending batch.")
        BATCH_STOP_SIGNAL_PATH.unlink()
        BATCH_ACTIVE_PATH.unlink(missing_ok=True)
        return

    next_entry = get_next_pending()
    if not next_entry:
        print("[BATCH] Queue empty — batch finished.")
        BATCH_ACTIVE_PATH.unlink(missing_ok=True)
        return

    try:
        # place_call() uses the synchronous `requests` library — run it
        # in a thread so a slow request doesn't block the event loop.
        await asyncio.to_thread(place_call, next_entry["phone"], get_exotel_settings())
        set_queue_status(next_entry["id"], "calling")
        print(f"[BATCH] Dialing next number: {next_entry['phone']}")
    except (ExotelConfigError, Exception) as e:
        print(f"[BATCH] Failed to dial next number: {e} — stopping batch.")
        BATCH_ACTIVE_PATH.unlink(missing_ok=True)


async def handle(ws):
    detector = None
    chat = None
    stream_sid = None
    customer_phone = None
    call_sample_rate = 8000
    call_errored = False
    _first_turn_was_slow = False
    _consecutive_gratitude_discards = 0

    # Set to a future timestamp every time Spark finishes speaking
    # (opening message AND every run_turn reply) — while time.time() is
    # still before this, incoming audio is dropped before it ever
    # reaches the turn detector. This deliberately covers BOTH echo
    # protection (Spark's own TTS leaking back right after it stops)
    # AND anything the customer said WHILE Spark was still talking —
    # earlier versions tried to capture and use that mid-speech audio,
    # but that meant responding to a "ghost" message from partway
    # through Spark's monologue while ignoring what the customer
    # actually meant to say once Spark finished. Simpler and more
    # correct to just discard everything until Spark is fully done,
    # then start listening fresh. The backlog of audio that queued up
    # DURING Spark's reply (we can't process new messages while
    # run_turn() is awaited) gets flushed past almost instantly once
    # the loop resumes — reading already-buffered bytes has no real-
    # time constraint — so this window only needs to be long enough to
    # cover that flush, not the length of Spark's whole reply.
    mute_until = 0.0
    MUTE_GRACE_SECONDS = 1.0

    _mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"[EXOTEL] Connection opened. [TIMING-DEBUG] wall clock: {time.strftime('%H:%M:%S.')}{int(time.time()*1000)%1000:03d} [MEM-DEBUG] RSS at connection start: {_mem_mb:.1f} MB")

    # Generous — during a genuinely active call, Exotel streams audio
    # continuously (every few hundred ms), so messages should never
    # stop arriving for anywhere close to this long. This is NOT
    # relying on websockets' own ping/pong keepalive (default ~20s
    # ping interval + ~20s ping timeout) or on the connection cleanly
    # closing — calls route through an ngrok tunnel, which sits
    # between Exotel and this server and can mask an upstream
    # disconnect, leaving the local connection looking alive with no
    # ConnectionClosed ever raised. This is an explicit, application-
    # level backstop that doesn't depend on the tunnel or library
    # defaults behaving as expected — the actual observed failure mode
    # this targets (a hang well past 30s with no ConnectionClosed).
    INACTIVITY_TIMEOUT_SECONDS = 45

    try:
        while True:
            try:
                raw_message = await asyncio.wait_for(ws.recv(), timeout=INACTIVITY_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                print(
                    f"[EXOTEL] No message received for {INACTIVITY_TIMEOUT_SECONDS}s "
                    f"— treating connection as dead and ending this call."
                )
                call_errored = True
                break

            event = json.loads(raw_message)
            event_type = event.get("event")

            if event_type == "connected":
                print("[EXOTEL] connected")

            elif event_type == "start":
                start_data = event.get("start", {})
                stream_sid = start_data.get("stream_sid")
                raw_phone = start_data.get("from")
                # Exotel reports this in domestic format ('09650707992')
                # while Call Queue stores E.164 ('+919650707992') — this
                # mismatch was the actual, confirmed reason numbers
                # never moved to "done"/"failed" in every test so far.
                # Normalize once, right here, so everything downstream
                # (finalize_call, mark_done_by_phone, mark_failed_by_phone)
                # uses the same format Call Queue does.
                customer_phone = normalize_phone(raw_phone) if raw_phone else raw_phone
                print(f"[EXOTEL] from={raw_phone!r} -> normalized to {customer_phone!r}")
                media_format = start_data.get("media_format", {})
                call_sample_rate = int(media_format.get("sample_rate", 8000))
                print(f"[EXOTEL] start — media_format={media_format}")

                # This machine's CPU has no real headroom for a second,
                # simultaneous conversation — Ollama/Whisper/Piper would
                # all be competing for the same limited resource,
                # degrading both calls rather than handling either one
                # well. Rather than silently accept a second call and
                # let both callers get a slow, degraded experience,
                # detect this up front (an existing call is already
                # tracked via ACTIVE_CALL_ID_PATH) and decline cleanly
                # with a short message instead. Deliberately checked
                # BEFORE creating a detector/ChatSession for this
                # connection — no need to spin up real resources for a
                # call we're about to end anyway.
                if ACTIVE_CALL_ID_PATH.exists():
                    print(
                        f"[EXOTEL] Already on a call — declining this one "
                        f"from {customer_phone!r} with a short message."
                    )
                    busy_text = "We will contact you soon. Regards, Smart Chip Tree."
                    pcm_bytes, native_rate = tts.synthesize_to_pcm(busy_text)
                    pcm_8k = resample_pcm16(pcm_bytes, native_rate, EXOTEL_SAMPLE_RATE).tobytes()
                    await send_pcm_as_media(ws, stream_sid, pcm_8k)
                    await asyncio.sleep(len(pcm_8k) / 2 / EXOTEL_SAMPLE_RATE + 0.5)

                    # Logged directly (not via finalize_call, which
                    # expects a real ChatSession) so a missed callback
                    # while busy is still visible in call history
                    # rather than silently vanishing — worth following
                    # up on manually, even though no conversation
                    # happened.
                    init_db()
                    save_call({
                        "call_id": uuid.uuid4().hex,
                        "phone": customer_phone,
                        "name": None,
                        "company": None,
                        "sector": None,
                        "requirements": None,
                        "quantity": None,
                        "brands": None,
                        "location": None,
                        "callback_time": None,
                        "status": "missed_busy",
                        "transcript": "[]",
                    })
                    break

                detector = CallTurnDetector(sample_rate=call_sample_rate, settings=VAD_SETTINGS)
                chat = ChatSession(llm=llm, knowledge=knowledge)

                # Lets Flask's live-transcript endpoint know which
                # transcripts/{id}.jsonl file to tail — that ChatSession
                # object lives in THIS process, not Flask's, so this is
                # the only way Flask finds out which call is active.
                ACTIVE_CALL_ID_PATH.parent.mkdir(exist_ok=True)
                ACTIVE_CALL_ID_PATH.write_text(chat.call_id)

                # Kick off the call directly via start_call_opening() —
                # NOT run_turn(ws, chat, stream_sid, ""), which routes
                # through send_streaming("") and hits its own "empty
                # input" guard ("Sorry, I didn't catch that") before
                # ever reaching the opening-message logic. This is a
                # plain deterministic reply (no LLM call involved), so
                # it's synthesized and sent directly rather than through
                # the background-thread/queue machinery run_turn() uses
                # for LLM-generated replies.
                opening = chat.start_call_opening()
                print(f"[SPARK] {opening}")
                pcm_bytes, native_rate = tts.synthesize_to_pcm(opening)
                pcm_8k = resample_pcm16(pcm_bytes, native_rate, EXOTEL_SAMPLE_RATE).tobytes()
                await send_pcm_as_media(ws, stream_sid, pcm_8k)
                # Scaled by the opening's own spoken duration, not just a
                # flat grace period — the opening is the longest single
                # utterance in the whole call, so it has the most time
                # for a customer to start talking partway through it,
                # meaning the largest possible backlog to flush before
                # live listening should resume. A fixed 1.0s grace was
                # observed letting speech from partway through a long
                # opening slip past the mute window and get treated as
                # live input.
                opening_seconds = len(pcm_8k) / 2 / EXOTEL_SAMPLE_RATE
                mute_until = time.time() + MUTE_GRACE_SECONDS + (opening_seconds * 0.15)

                # Tracks genuine customer silence — distinct from
                # INACTIVITY_TIMEOUT_SECONDS above, which only fires if
                # messages stop arriving AT ALL (a dead connection).
                # Exotel streams audio continuously even while the
                # customer says nothing, so that timeout alone would
                # never catch "still connected, but not responding" —
                # a muted mic, background noise masking their voice, or
                # just not answering. last_activity_at updates on every
                # genuine customer turn; if too much time passes without
                # one, proactively check in rather than leaving them in
                # silence, then actually end the call if that also gets
                # no response.
                last_activity_at = time.time()
                silence_prompted = False
                SILENCE_PROMPT_SECONDS = 10
                SILENCE_HANGUP_SECONDS = 20

            elif event_type == "media":
                if detector is None:
                    continue
                if time.time() < mute_until:
                    # Spark is speaking, or just finished too recently
                    # — drop this rather than accumulate or act on it,
                    # whether it's the customer talking mid-monologue,
                    # backlog flushing through, or actual TTS echo.
                    continue

                # Keep last_activity_at fresh whenever the customer is
                # actively mid-utterance, not only once a turn fully
                # completes. Without this, a long, slow, or naturally
                # pause-heavy answer that takes more than
                # SILENCE_PROMPT_SECONDS to finish could get wrongly
                # interrupted by the "are you still there?" check-in
                # mid-sentence — confirmed directly: the VAD was one
                # silence frame away from closing out a real turn when
                # the check-in fired anyway, spoke over the customer,
                # and the resulting mute window then dropped exactly
                # the frames needed to finish, restarting the cycle.
                # detector.triggered reflects state as of the last
                # PROCESSED frame, so checking it here (before this
                # frame is processed) still correctly reflects whether
                # they were actively talking a moment ago.
                if detector.triggered:
                    last_activity_at = time.time()
                    silence_prompted = False

                silence_elapsed = time.time() - last_activity_at
                if silence_elapsed > SILENCE_HANGUP_SECONDS:
                    print(
                        f"[EXOTEL] No response for {SILENCE_HANGUP_SECONDS}s "
                        f"after the check-in prompt — ending call as unresponsive."
                    )
                    reply_text = "I'm having trouble hearing you, so I'll let you go for now. Feel free to call back anytime — have a great day!"
                    print(f"[SPARK] {reply_text}")
                    pcm_bytes, native_rate = tts.synthesize_to_pcm(reply_text)
                    pcm_8k = resample_pcm16(pcm_bytes, native_rate, EXOTEL_SAMPLE_RATE).tobytes()
                    await send_pcm_as_media(ws, stream_sid, pcm_8k)
                    await asyncio.sleep(len(pcm_8k) / 2 / EXOTEL_SAMPLE_RATE + 0.5)
                    break
                elif silence_elapsed > SILENCE_PROMPT_SECONDS and not silence_prompted:
                    silence_prompted = True
                    print(f"[EXOTEL] {SILENCE_PROMPT_SECONDS}s of silence — proactively checking in.")
                    reply_text = "Sorry, I'm not able to hear you — are you still there?"
                    print(f"[SPARK] {reply_text}")
                    pcm_bytes, native_rate = tts.synthesize_to_pcm(reply_text)
                    pcm_8k = resample_pcm16(pcm_bytes, native_rate, EXOTEL_SAMPLE_RATE).tobytes()
                    await send_pcm_as_media(ws, stream_sid, pcm_8k)
                    prompt_seconds = len(pcm_8k) / 2 / EXOTEL_SAMPLE_RATE
                    mute_until = time.time() + MUTE_GRACE_SECONDS + (prompt_seconds * 0.15)
                    continue

                payload = event["media"]["payload"]
                turn_audio = detector.add_media_payload(payload)

                if turn_audio is not None:
                    # Sent immediately, before STT even starts — this
                    # is the exact moment a genuine speech turn was
                    # confirmed (already past add_media_payload's own
                    # min-speech-length and min-RMS false-trigger
                    # filters), so this won't fire on background noise
                    # VAD mistakenly triggered on. Covers the silent
                    # STT window the existing stall filler never
                    # reached — see the pre-synthesis comment at
                    # startup for the full reasoning.
                    await send_pcm_as_media(ws, stream_sid, ACKNOWLEDGMENT_PCM_8K)
                    mute_until = time.time() + MUTE_GRACE_SECONDS + (ACKNOWLEDGMENT_SECONDS * 0.15)

                    duration_s = len(turn_audio) / 2 / call_sample_rate
                    padding_s = (VAD_SETTINGS["vad_preroll_ms"] + VAD_SETTINGS["vad_silence_ms"]) / 1000
                    estimated_speech_s = max(0.0, duration_s - padding_s)
                    # Actual RMS of this turn's audio, not just whether
                    # it passed the min_rms floor — needed to tell apart
                    # faint echo/noise (RMS sitting just above the
                    # threshold) from something else entirely (normal
                    # speech-level RMS despite no one having spoken),
                    # which point to genuinely different root causes and
                    # need different fixes.
                    turn_rms = _compute_rms(turn_audio)

                    resampled = resample_pcm16(turn_audio, call_sample_rate, WHISPER_SAMPLE_RATE)
                    whisper_input = pcm16_to_whisper_input(resampled)

                    result = stt.transcribe_array(whisper_input, estimated_speech_s)
                    timestamp = time.strftime("%H:%M:%S")
                    logprob = result.get("avg_logprob")
                    logprob_str = f"{logprob:.3f}" if logprob is not None else "N/A"
                    stt_duration = result.get("duration")
                    stt_duration_str = f"{stt_duration:.2f}s" if stt_duration is not None else "N/A"
                    print(
                        f"[CUSTOMER @ {timestamp}] {result['text']!r} "
                        f"(model={result['model_used']}, rms={turn_rms:.1f}, "
                        f"avg_logprob={logprob_str}, stt_time={stt_duration_str})"
                    )

                    # A bare "thank you"/"thanks" is a well-documented
                    # Whisper hallucination on noise or silence — its
                    # training data is YouTube-heavy, where videos
                    # commonly end with "thank you for watching", giving
                    # it a strong bias toward this exact phrase on
                    # ambiguous audio. Direct testing confirmed a single
                    # cough or loud breath reliably produces this, at
                    # RMS levels (1000+) that genuinely overlap with real
                    # quiet speech — so raising the RMS floor further
                    # can't fix this without also rejecting real speech.
                    # Given how asymmetric the cost is (occasionally
                    # missing a genuine "thank you" vs. the call
                    # derailing into an infinite loop), treat this as
                    # equivalent to hearing nothing: discard before it
                    # ever reaches chat_session, don't respond, keep
                    # listening.
                    bare_check = result["text"].strip().lower().rstrip(".,!?")
                    if bare_check in GRATITUDE_PHRASES:
                        print(f"[FILTER] Discarding likely-hallucinated {result['text']!r} (known Whisper noise artifact)")
                    elif logprob is not None and logprob < CONFIDENCE_REJECT_THRESHOLD:
                        # Whisper itself wasn't confident about this
                        # transcription — real data showed confirmed
                        # gibberish clustering well below this
                        # threshold while genuine speech stayed above
                        # it. Ask the customer to repeat rather than
                        # confidently acting on a likely-wrong guess.
                        # Deterministic, no LLM call — synthesized and
                        # sent directly, same pattern as the opening
                        # message, since respond_low_confidence() is a
                        # fixed reply with nothing to stream.
                        print(f"[FILTER] Low confidence (avg_logprob={logprob:.3f}) — asking customer to repeat")
                        reply_text = chat.respond_low_confidence()
                        print(f"[SPARK] {reply_text}")
                        pcm_bytes, native_rate = tts.synthesize_to_pcm(reply_text)
                        pcm_8k = resample_pcm16(pcm_bytes, native_rate, EXOTEL_SAMPLE_RATE).tobytes()
                        await send_pcm_as_media(ws, stream_sid, pcm_8k)
                        mute_until = time.time() + MUTE_GRACE_SECONDS
                        last_activity_at = time.time()
                        silence_prompted = False
                    elif result["text"].strip():
                        _turn_start = time.perf_counter()
                        print(f"[TIMING-DEBUG] run_turn() starting — wall clock: {time.strftime('%H:%M:%S.')}{int(time.time()*1000)%1000:03d}")
                        audio_seconds = await run_turn(ws, chat, stream_sid, result["text"])
                        print(f"[TIMING-DEBUG] run_turn() total wall time: {time.perf_counter() - _turn_start:.3f}s")
                        _mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                        print(f"[MEM-DEBUG] this process RSS after turn: {_mem_mb:.1f} MB")

                        # Checked specifically on the call's FIRST real
                        # turn — this is the exact signal confirmed to
                        # correlate with this process's growing memory
                        # footprint (see FIRST_TURN_EVAL_RESTART_THRESHOLD_SECONDS
                        # above). Not checked on later turns, since the
                        # knowledge-injection turns have their own,
                        # separate, already-diagnosed cost that
                        # shouldn't get conflated with this signal.
                        if chat.exchange_count == 1 and chat.last_prompt_eval_seconds is not None:
                            if chat.last_prompt_eval_seconds > FIRST_TURN_EVAL_RESTART_THRESHOLD_SECONDS:
                                _first_turn_was_slow = True
                                print(
                                    f"[RESTART-DEBUG] Turn 1 prompt-eval "
                                    f"({chat.last_prompt_eval_seconds:.2f}s) exceeded "
                                    f"the {FIRST_TURN_EVAL_RESTART_THRESHOLD_SECONDS}s "
                                    f"threshold — this process will restart once "
                                    f"this call ends."
                                )
                        # Same scaling as the opening message — a longer
                        # reply means more time for a larger backlog to
                        # accumulate if the customer starts talking
                        # partway through it.
                        mute_until = time.time() + MUTE_GRACE_SECONDS + (audio_seconds * 0.15)
                        last_activity_at = time.time()
                        silence_prompted = False

                        if chat.call_finished:
                            # run_turn() only waits for audio to be SENT
                            # (a fast network write), not for it to
                            # actually finish PLAYING on the customer's
                            # phone — those are very different timescales.
                            # Closing right after send() previously cut
                            # the final message off mid-sentence. Wait
                            # for the real playback duration (plus a
                            # small buffer) before actually hanging up.
                            print(
                                f"[EXOTEL] Call finished — waiting "
                                f"{audio_seconds:.1f}s for final audio "
                                f"to finish playing."
                            )
                            await asyncio.sleep(audio_seconds + 0.5)
                            print("[EXOTEL] Closing connection.")
                            await ws.close()
                            break

            elif event_type == "stop":
                print("[EXOTEL] stop — call ended.")
                break

    except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError, asyncio.TimeoutError) as e:
        # An unexpected disconnect (network drop, Exotel ending the call
        # from their side, etc) should end THIS call, not crash the
        # whole server process — every other call and the loaded models
        # stay alive for the next one. asyncio.TimeoutError specifically
        # comes from send_pcm_as_media's internal timeout — a hang
        # there means the connection is effectively dead even without
        # an explicit ConnectionClosed having been raised yet.
        print(f"[EXOTEL] Connection error, ending this call: {e}")
        call_errored = True

    finally:
        # Runs no matter which of the three exit paths triggered (clean
        # auto-end, Exotel's own "stop" event, or a connection error) —
        # every real call gets saved exactly once, instead of only the
        # ones that happened to end "badly" (the same class of bug the
        # local CLI version had early on, before it called this
        # unconditionally too). chat is None if the connection dropped
        # before the "start" event ever arrived — nothing to save yet.
        if chat is not None:
            # simulate_exotel_call.py connects to this exact, real
            # server (that's the whole point of it) - it always
            # identifies itself with this same, fixed fake number,
            # which no real caller would ever have. Detected here so
            # simulator runs don't pollute the real call_queue/dataset
            # the same way local-mic tests were fixed to not do
            # earlier - redirected for just this one finalize_call()
            # call, restored immediately after, not left changed for
            # the rest of this process's real, ongoing calls.
            is_simulated_call = customer_phone == "+919999999999"
            if is_simulated_call:
                _original_db_path = call_storage.DB_PATH
                _original_transcript_dir = chat_session.TRANSCRIPT_DIR
                call_storage.DB_PATH = Path("data/spark_local_test.db")
                call_storage.TRANSCRIPT_DIR = Path("transcripts_local_test")
                chat_session.TRANSCRIPT_DIR = Path("transcripts_local_test")
                chat_session.TRANSCRIPT_DIR.mkdir(exist_ok=True)

            try:
                _cleanup_start = time.perf_counter()
                finalize_call(chat, phone=customer_phone)
                print(f"[TIMING-DEBUG] finalize_call() took {time.perf_counter() - _cleanup_start:.3f}s")
            finally:
                if is_simulated_call:
                    call_storage.DB_PATH = _original_db_path
                    call_storage.TRANSCRIPT_DIR = _original_transcript_dir
                    chat_session.TRANSCRIPT_DIR = _original_transcript_dir

            # Counts toward the restart threshold below only if this
            # was a genuine, real conversation — not a missed_busy
            # rejection (those never even create a ChatSession, so
            # chat is None and this block never runs for them) and not
            # a zero-interaction hangup (connects, plays the opening,
            # then ends with no customer turn at all).
            if chat.exchange_count > 0:
                global _completed_call_count
                _completed_call_count += 1
                print(f"[RESTART-DEBUG] Completed calls this session: {_completed_call_count}/{MAX_CALLS_BEFORE_RESTART}")

            _mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            print(f"[MEM-DEBUG] RSS at call end: {_mem_mb:.1f} MB")

            # Call genuinely over — clear the "live" marker so Flask's
            # live-transcript endpoint correctly reports nothing in
            # progress, rather than continuing to point at a finished
            # call's transcript file.
            ACTIVE_CALL_ID_PATH.unlink(missing_ok=True)

            if customer_phone:
                mark_status = mark_failed_by_phone if call_errored else mark_done_by_phone
                rows_updated = mark_status(customer_phone)
                if rows_updated == 0:
                    print(
                        f"[BATCH] WARNING: no Call Queue entry matched "
                        f"phone={customer_phone!r} — likely a format "
                        f"mismatch against what's stored there. Check "
                        f"the [EXOTEL] from= line above against the "
                        f"number shown in Call Queue."
                    )

            # Restart decision happens BEFORE continuing the batch,
            # deliberately — dialing a fresh customer and then
            # immediately tearing down the server for a full model
            # reload would connect them to a server that's disappearing
            # or already gone. If a restart is needed, this cycle skips
            # dialing entirely and leaves the queue as-is; the freshly
            # restarted process resumes it on startup (see main()).
            #
            # Confirmed via direct measurement (gc.collect() froze at
            # 0.0 MB freed across two real calls, despite genuinely
            # collecting hundreds of garbage objects each time): this
            # machine's process memory footprint grows call over call
            # not because of a reference leak in our own code, but
            # because CPython's allocator doesn't hand freed memory
            # back to the OS. That growing footprint measurably
            # correlates with slower first-turn prompt-eval on the
            # NEXT call (1061MB->1.88s vs 1345MB->9.39s, confirmed
            # directly) - likely because it pressures Windows into
            # evicting some of Ollama's memory-mapped model-weight
            # pages, which then have to be re-read from disk. A clean
            # process restart is the only reliable way to actually
            # return this memory - there's no Windows equivalent to
            # Linux's malloc_trim() that would reclaim it without one.
            needs_restart = _first_turn_was_slow or _completed_call_count >= MAX_CALLS_BEFORE_RESTART

            if needs_restart:
                reason = (
                    "turn 1's prompt-eval crossed the threshold this call"
                    if _first_turn_was_slow
                    else f"reached the {MAX_CALLS_BEFORE_RESTART}-call safety-net limit"
                )
                print(
                    f"[RESTART] Restarting the process ({reason}) — releasing "
                    f"accumulated memory before it affects the next call's "
                    f"latency. The launcher script should relaunch "
                    f"automatically. Skipping this cycle's dial — the "
                    f"restarted process will resume the batch on startup."
                )
                _shutdown_event.set()
            elif BATCH_ACTIVE_PATH.exists():
                # Only continue the chain if THIS call was itself part
                # of an active batch — a one-off manual "Call" click
                # from Call Queue's per-row button must NOT cascade
                # into dialing the rest of the queue.
                await _continue_batch()


async def main():
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    print("Two-way pipeline test server listening on ws://0.0.0.0:5001")
    print("Run 'ngrok http 5001' (or cloudflared) in another terminal to expose this publicly.")
    async with websockets.serve(handle, "0.0.0.0", 5001):
        # If a batch was still active when this process restarted (the
        # normal, expected reason for a restart mid-batch — see the
        # restart-decision comment above), nothing else would ever
        # resume it: _continue_batch() only ever gets triggered by "a
        # call just ended", and after a restart there's no such event
        # to trigger it. Checked here, after the server is already
        # listening, so it's actually ready for the connection Exotel
        # makes back once place_call() dials.
        if BATCH_ACTIVE_PATH.exists():
            print("[RESTART] A batch was active before this restart — resuming it now.")
            await _continue_batch()

        await _shutdown_event.wait()
    print("[RESTART] Server closed. Exiting process for the launcher to relaunch.")


if __name__ == "__main__":
    asyncio.run(main())