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
import json
import queue
import time
from pathlib import Path

import websockets

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
from app.call_storage import mark_done_by_phone, mark_failed_by_phone, get_next_pending, set_queue_status

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
PAUSE_BETWEEN_CALLS_SECONDS = 150  # 2.5 minutes

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
print("Shared resources ready.")


async def send_pcm_as_media(ws, stream_sid, pcm_bytes):
    """Splits PCM audio into Exotel-sized chunks and sends each as its
    own media event."""
    for i in range(0, len(pcm_bytes), OUTGOING_CHUNK_BYTES):
        chunk = pcm_bytes[i:i + OUTGOING_CHUNK_BYTES]
        payload = base64.b64encode(chunk).decode()
        await ws.send(json.dumps({
            "event": "media",
            "stream_sid": stream_sid,
            "media": {"payload": payload},
        }))


async def run_turn(ws, chat, stream_sid, user_message):
    """
    Runs one full Spark turn: chat_session.send_streaming() in a
    background thread (it's synchronous), streaming each sentence's
    synthesized audio back over the WebSocket as it becomes ready
    rather than waiting for the entire reply to finish.
    """
    audio_queue = queue.Queue()
    SENTINEL = object()

    def on_sentence(sentence_text):
        print(f"[SPARK] {sentence_text}")
        pcm_bytes, native_rate = tts.synthesize_to_pcm(sentence_text)
        pcm_8k = resample_pcm16(pcm_bytes, native_rate, EXOTEL_SAMPLE_RATE).tobytes()
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

    print("[EXOTEL] Connection opened.")

    try:
        async for raw_message in ws:
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
                mute_until = time.time() + MUTE_GRACE_SECONDS

            elif event_type == "media":
                if detector is None:
                    continue
                if time.time() < mute_until:
                    # Spark is speaking, or just finished too recently
                    # — drop this rather than accumulate or act on it,
                    # whether it's the customer talking mid-monologue,
                    # backlog flushing through, or actual TTS echo.
                    continue
                payload = event["media"]["payload"]
                turn_audio = detector.add_media_payload(payload)

                if turn_audio is not None:
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
                    print(
                        f"[CUSTOMER @ {timestamp}] {result['text']!r} "
                        f"(model={result['model_used']}, rms={turn_rms:.1f}, avg_logprob={logprob_str})"
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
                    elif result["text"].strip():
                        await run_turn(ws, chat, stream_sid, result["text"])
                        mute_until = time.time() + MUTE_GRACE_SECONDS

                        if chat.call_finished:
                            # Mirrors what the local CLI (main.py)
                            # already does — the call is genuinely over
                            # (auto-end fast path fired). Close it from
                            # our side instead of continuing to listen,
                            # which previously meant crashing later
                            # when Exotel's own side eventually killed
                            # the connection out from under us mid-send.
                            print("[EXOTEL] Call finished — closing connection.")
                            await ws.close()
                            break

            elif event_type == "stop":
                print("[EXOTEL] stop — call ended.")
                break

    except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError) as e:
        # An unexpected disconnect (network drop, Exotel ending the call
        # from their side, etc) should end THIS call, not crash the
        # whole server process — every other call and the loaded models
        # stay alive for the next one.
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
            finalize_call(chat, phone=customer_phone)

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

            # Only continue the chain if THIS call was itself part of
            # an active batch — a one-off manual "Call" click from
            # Call Queue's per-row button must NOT cascade into dialing
            # the rest of the queue.
            if BATCH_ACTIVE_PATH.exists():
                await _continue_batch()


async def main():
    print("Two-way pipeline test server listening on ws://0.0.0.0:5001")
    print("Run 'ngrok http 5001' (or cloudflared) in another terminal to expose this publicly.")
    async with websockets.serve(handle, "0.0.0.0", 5001):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())