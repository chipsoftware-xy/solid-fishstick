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
import threading
import time
import uuid
import wave
from pathlib import Path

import psutil
import numpy as np
import websockets

from telephony.stream_vad import CallTurnDetector, _compute_rms
from telephony.interruption import should_interrupt
from telephony.audio_utils import resample_pcm16, pcm16_to_whisper_input
from app.stt.deepinfra_stt import STT as DeepInfraSTT, NAME_HINT_PROMPT
from app.llm.chat_session import GRATITUDE_PHRASES
from telephony.dialer import place_call, get_exotel_settings, ExotelConfigError, normalize_phone
from app.tts.piper import TTS
from app.llm.chat_session import ChatSession, NO_RESPONSE_SENTINEL
from app.llm.deepinfra_client import LLM
from app.knowledge_base import KnowledgeBase
from app.call_finalizer import finalize_call
from app.call_storage import mark_done_by_phone, mark_failed_by_phone, get_next_pending, set_queue_status, save_call, init_db

import app.llm.chat_session as chat_session
import app.call_storage as call_storage

VAD_SETTINGS = {
    "vad_aggressiveness": 2,
    "vad_frame_ms": 30,
    "vad_preroll_ms": 300,
    
    "vad_silence_ms": 1200,
    "vad_min_speech_ms": 300,
    "vad_max_utterance_ms": 20000,
    
    "vad_min_rms": 700,
}


INTERRUPTION_VAD_SETTINGS = {**VAD_SETTINGS, "vad_silence_ms": 500}

WHISPER_SAMPLE_RATE = 16000

DEBUG_AUDIO_DIR = Path("debug_audio")
DEBUG_AUDIO_DIR.mkdir(exist_ok=True)


def _save_debug_wav(pcm16_bytes, sample_rate, label):
    filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{label}.wav"
    filepath = DEBUG_AUDIO_DIR / filename
    with wave.open(str(filepath), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # PCM16 = 2 bytes/sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_bytes)
    return filepath

MAX_CALLS_BEFORE_RESTART = 20
_completed_call_count = 0
_shutdown_event = None  # created inside main(), since it needs a running event loop
# Within Exotel's documented 3200-100000 byte range, a multiple of 320.
# 3200 bytes at 8kHz/16-bit mono = 200ms of audio per outgoing chunk.
OUTGOING_CHUNK_BYTES = 3200


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
stt = DeepInfraSTT()
tts = TTS()
llm = LLM()
knowledge = KnowledgeBase()


asyncio.run(stt.verify())

print("Shared resources ready.")


SEND_TIMEOUT_SECONDS = 30


async def send_pcm_as_media(ws, stream_sid, pcm_bytes, sample_rate, interrupted=None):
   
    chunk_duration_seconds = OUTGOING_CHUNK_BYTES / 2 / sample_rate

    async def _send():
        for i in range(0, len(pcm_bytes), OUTGOING_CHUNK_BYTES):
            if interrupted is not None and interrupted.is_set():
                return
            chunk = pcm_bytes[i:i + OUTGOING_CHUNK_BYTES]
            payload = base64.b64encode(chunk).decode()
            await ws.send(json.dumps({
                "event": "media",
                "stream_sid": stream_sid,
                "media": {"payload": payload},
            }))
            await asyncio.sleep(chunk_duration_seconds)

    await asyncio.wait_for(_send(), timeout=SEND_TIMEOUT_SECONDS)


async def speak(ws, stream_sid, pcm_bytes, sample_rate, spark_speaking, spark_state=None, text=""):
    
    if spark_state is not None:
        spark_state["current_text"] = text
    spark_speaking.set()
    try:
        await send_pcm_as_media(ws, stream_sid, pcm_bytes, sample_rate)
    finally:
        spark_speaking.clear()


async def run_turn(ws, chat, stream_sid, user_message, sample_rate, spark_speaking=None, spark_state=None, interrupted=None, call_ended=None):
    
    audio_queue = queue.Queue()
    SENTINEL = object()
    total_audio_seconds = 0.0
    # threading.Event, not asyncio.Event — run_chat() executes in a
    # separate OS thread (via asyncio.to_thread), same reasoning as
    # call_ended elsewhere in this file: a thread needs a thread-safe
    # signaling primitive, not an asyncio one.
    generation_complete = threading.Event()

    def on_sentence(sentence_text):
        nonlocal total_audio_seconds
        if call_ended is not None and call_ended.is_set():
            return
        print(f"[SPARK] {sentence_text}")
        if spark_state is not None:
            spark_state["current_text"] = sentence_text
        _synth_start = time.perf_counter()
        pcm_bytes, native_rate = tts.synthesize_to_pcm(sentence_text)
        _synth_seconds = time.perf_counter() - _synth_start
        pcm_8k = resample_pcm16(pcm_bytes, native_rate, sample_rate).tobytes()
        # PCM16 = 2 bytes per sample.
        _sentence_audio_seconds = len(pcm_8k) / 2 / sample_rate
        total_audio_seconds += _sentence_audio_seconds
        print(f"[TIMING-SENTENCE] synth={_synth_seconds:.2f}s audio_duration={_sentence_audio_seconds:.2f}s words={len(sentence_text.split())}")
        audio_queue.put(pcm_8k)

    def run_chat():
        try:
            chat.send_streaming(user_message, on_sentence)
        finally:
            
            generation_complete.set()
            audio_queue.put(SENTINEL)

    thread_task = asyncio.create_task(asyncio.to_thread(run_chat))

    if spark_speaking is not None:
        spark_speaking.set()

    try:
        while True:
            if interrupted is not None and interrupted.is_set():
               
                break
            _queue_wait_start = time.perf_counter()
            item = await asyncio.to_thread(audio_queue.get)
            _queue_wait_seconds = time.perf_counter() - _queue_wait_start
            if item is SENTINEL:
                break
            if _queue_wait_seconds > 0.5:
               
                print(f"[TIMING-SENTENCE] queue_wait={_queue_wait_seconds:.2f}s (gap before this sentence was ready)")

            if generation_complete.is_set() and audio_queue.qsize() <= 1:
               
                asyncio.create_task(_fire_stt_warmup_ping())

            await send_pcm_as_media(ws, stream_sid, item, sample_rate, interrupted=interrupted)
    finally:
        if spark_speaking is not None:
            spark_speaking.clear()

    await thread_task
    return total_audio_seconds


async def _fire_stt_warmup_ping():
    
    try:
        warmup_clip = np.zeros(int(0.3 * stt.sample_rate), dtype=np.float32)
        await stt.transcribe_array(warmup_clip, duration_s=0.3)
    except Exception as e:
        print(f"[WARN] STT warm-up ping failed (non-fatal, ignored): {e}")


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
    _consecutive_gratitude_discards = 0
  
    spark_speaking = asyncio.Event()
   
    interrupted = asyncio.Event()
   
    pending_interruption_text = None
  
    spark_state = {"current_text": ""}
  
    interruption_detector = None
    
    pending_interruption_check_task = None
   
    _spark_was_speaking_last_frame = False
   
    speech_task = None
 
    call_ended = threading.Event()

    _mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"[EXOTEL] Connection opened. [TIMING-DEBUG] wall clock: {time.strftime('%H:%M:%S.')}{int(time.time()*1000)%1000:03d} [MEM-DEBUG] RSS at connection start: {_mem_mb:.1f} MB")

   
    INACTIVITY_TIMEOUT_SECONDS = 45

    async def start_new_turn(turn_text, previous_task):
       
        nonlocal last_activity_at, silence_prompted, pending_interruption_text, speech_task
        try:
            if previous_task is not None and not previous_task.done():
                await previous_task

            current_text = turn_text
            _turn_start = time.perf_counter()

            while True:
                interrupted.clear()
                audio_seconds = await run_turn(ws, chat, stream_sid, current_text, call_sample_rate, spark_speaking=spark_speaking, spark_state=spark_state, interrupted=interrupted, call_ended=call_ended)
                print(f"[TIMING-DEBUG] run_turn() total wall time: {time.perf_counter() - _turn_start:.3f}s")
                _mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                print(f"[MEM-DEBUG] this process RSS after turn: {_mem_mb:.1f} MB")

                if interrupted.is_set() and pending_interruption_text:
                    print(f"[BARGE-IN] Processing interruption as new turn: {pending_interruption_text!r}")
                    current_text = pending_interruption_text
                    pending_interruption_text = None
                    continue

                break

            if call_ended.is_set():
                return

            last_activity_at = time.time()
            silence_prompted = False

            if chat.call_finished:
                print(
                    f"[EXOTEL] Call finished — waiting "
                    f"{audio_seconds:.1f}s for final audio "
                    f"to finish playing."
                )
                await asyncio.sleep(audio_seconds + 0.5)
                print("[EXOTEL] Closing connection.")
                await ws.close()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[WARN] start_new_turn failed with an unexpected error, attempting a fallback reply rather than going silent: {e!r}")
            if not call_ended.is_set():
                try:
                    fallback_text = "Sorry, I didn't quite catch that — could you say that again?"
                    print(f"[SPARK] {fallback_text}")
                    fallback_pcm_bytes, fallback_native_rate = tts.synthesize_to_pcm(fallback_text)
                    fallback_pcm_8k = resample_pcm16(fallback_pcm_bytes, fallback_native_rate, call_sample_rate).tobytes()
                    await speak(ws, stream_sid, fallback_pcm_8k, call_sample_rate, spark_speaking, spark_state=spark_state, text=fallback_text)
                except Exception as fallback_error:
                    print(f"[WARN] Fallback reply itself also failed, giving up on this turn: {fallback_error!r}")

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
                
                customer_phone = normalize_phone(raw_phone) if raw_phone else raw_phone
                print(f"[EXOTEL] from={raw_phone!r} -> normalized to {customer_phone!r}")
                media_format = start_data.get("media_format", {})
                call_sample_rate = int(media_format.get("sample_rate", 8000))
                print(f"[EXOTEL] start — media_format={media_format}")

              
                if ACTIVE_CALL_ID_PATH.exists():
                    print(
                        f"[EXOTEL] Already on a call — declining this one "
                        f"from {customer_phone!r} with a short message."
                    )
                    busy_text = "We will contact you soon. Regards, Smart Chip Tree."
                    pcm_bytes, native_rate = tts.synthesize_to_pcm(busy_text)
                    pcm_8k = resample_pcm16(pcm_bytes, native_rate, call_sample_rate).tobytes()
                    await speak(ws, stream_sid, pcm_8k, call_sample_rate, spark_speaking, spark_state=spark_state, text=busy_text)
                    await asyncio.sleep(len(pcm_8k) / 2 / call_sample_rate + 0.5)

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
                interruption_detector = CallTurnDetector(sample_rate=call_sample_rate, settings=INTERRUPTION_VAD_SETTINGS)
                chat = ChatSession(llm=llm, knowledge=knowledge)

              
                ACTIVE_CALL_ID_PATH.parent.mkdir(exist_ok=True)
                ACTIVE_CALL_ID_PATH.write_text(chat.call_id)

                
                opening = chat.start_call_opening()
                print(f"[SPARK] {opening}")
                pcm_bytes, native_rate = tts.synthesize_to_pcm(opening)
                pcm_8k = resample_pcm16(pcm_bytes, native_rate, call_sample_rate).tobytes()

            
                _normal_silence_frames_needed = detector.silence_frames_needed
                detector.silence_frames_needed = max(1, int(500 / detector.frame_ms))

                
                opening_turn_claimed = {"value": False}

                async def _advance_opening_after_fixed_delay():
                    await asyncio.sleep(1.0)
                    if opening_turn_claimed["value"]:
                        # The main detector below already captured a
                        # genuine, quick customer response (e.g. an
                        # explicit rejection) and claimed this turn
                        # first — don't also advance here on top of it.
                        return
                    opening_turn_claimed["value"] = True
                    detector.silence_frames_needed = _normal_silence_frames_needed
                    nonlocal speech_task
                    previous = speech_task
                    speech_task = asyncio.create_task(start_new_turn("Okay", previous))

                asyncio.create_task(_advance_opening_after_fixed_delay())

               
                async def _speak_opening():
                    try:
                        await speak(ws, stream_sid, pcm_8k, call_sample_rate, spark_speaking, spark_state=spark_state, text=opening)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        # Same reasoning as _speak_reply's own guard —
                        # an abandoned task's exception becoming
                        # unhandled achieves nothing useful, whether the
                        # call has already ended or is still going.
                        print(f"[WARN] _speak_opening failed with an unexpected error, recovering rather than crashing: {e!r}")

                speech_task = asyncio.create_task(_speak_opening())

               
                last_activity_at = time.time()
                silence_prompted = False
                SILENCE_PROMPT_SECONDS = 10
                SILENCE_HANGUP_SECONDS = 20

            elif event_type == "media":
                if detector is None:
                    continue

                if spark_speaking.is_set():
                   
                    last_activity_at = time.time()

                    interruption_payload = event["media"]["payload"]
                    interruption_turn_audio = interruption_detector.add_media_payload(interruption_payload)

                    if interruption_turn_audio is not None:
                      
                        async def _transcribe_and_decide(audio_for_stt, spark_text_at_capture_time):
                            try:
                                interruption_raw_wav_path = _save_debug_wav(audio_for_stt, call_sample_rate, "interruption_raw8k")

                                interruption_resampled = resample_pcm16(audio_for_stt, call_sample_rate, WHISPER_SAMPLE_RATE)
                                interruption_whisper_input = pcm16_to_whisper_input(interruption_resampled)
                                interruption_result = await stt.transcribe_array(
                                    interruption_whisper_input, 0.0,
                                    prompt=NAME_HINT_PROMPT if chat.waiting_for_name else None,
                                )
                                candidate_text = interruption_result["text"].strip()
                                print(f"[DEBUG-AUDIO] interruption raw={interruption_raw_wav_path.name}")

                                if should_interrupt(candidate_text, spark_text_at_capture_time):
                                    print(f"[BARGE-IN] Genuine interruption detected: {candidate_text!r} (Spark was saying: {spark_text_at_capture_time!r})")
                                    if spark_speaking.is_set():
                                        
                                        nonlocal pending_interruption_text
                                        pending_interruption_text = candidate_text
                                        interrupted.set()
                                    else:
                                        
                                       
                                        if chat.call_finished:
                                            print(f"[BARGE-IN] Call already finished via the normal turn path — not starting a duplicate turn for {candidate_text!r}.")
                                        else:
                                            nonlocal speech_task
                                            previous = speech_task
                                            speech_task = asyncio.create_task(start_new_turn(candidate_text, previous))
                                else:
                                    print(f"[BARGE-IN] Ignored (filler or echo): {candidate_text!r}")
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                print(f"[WARN] _transcribe_and_decide failed with an unexpected error, recovering rather than crashing: {e!r}")

                        if pending_interruption_check_task is not None and not pending_interruption_check_task.done():
                            
                            pending_interruption_check_task.cancel()
                        pending_interruption_check_task = asyncio.create_task(_transcribe_and_decide(interruption_turn_audio, spark_state["current_text"]))

                    _spark_was_speaking_last_frame = True
                    continue

                if _spark_was_speaking_last_frame and interruption_detector.triggered:
                    
                    print("[BARGE-IN] Handing off interruption detector's in-progress utterance to the main detector.")
                    detector.triggered = interruption_detector.triggered
                    detector.speech_frames = interruption_detector.speech_frames
                    detector.silence_run = interruption_detector.silence_run
                    detector.speech_frame_count = interruption_detector.speech_frame_count
                    interruption_detector._reset_state()

                _spark_was_speaking_last_frame = False

                payload = event["media"]["payload"]

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
                    pcm_8k = resample_pcm16(pcm_bytes, native_rate, call_sample_rate).tobytes()
                    await speak(ws, stream_sid, pcm_8k, call_sample_rate, spark_speaking, spark_state=spark_state, text=reply_text)
                    await asyncio.sleep(len(pcm_8k) / 2 / call_sample_rate + 0.5)
                    break
                elif silence_elapsed > SILENCE_PROMPT_SECONDS and not silence_prompted:
                    silence_prompted = True
                    print(f"[EXOTEL] {SILENCE_PROMPT_SECONDS}s of silence — proactively checking in.")
                    reply_text = "Sorry, I'm not able to hear you — are you still there?"
                    print(f"[SPARK] {reply_text}")
                    pcm_bytes, native_rate = tts.synthesize_to_pcm(reply_text)
                    pcm_8k = resample_pcm16(pcm_bytes, native_rate, call_sample_rate).tobytes()
                    async def _speak_checkin():
                        try:
                            await speak(ws, stream_sid, pcm_8k, call_sample_rate, spark_speaking, spark_state=spark_state, text=reply_text)
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(f"[WARN] _speak_checkin failed with an unexpected error, recovering rather than crashing: {e!r}")
                    speech_task = asyncio.create_task(_speak_checkin())
                    continue

                turn_audio = detector.add_media_payload(payload)

                if turn_audio is not None:
                    if chat.waiting_for_opening_pitch:
                        if opening_turn_claimed["value"]:
                            
                            print("[TIMING-DEBUG] Opening turn already claimed by the fixed-delay advance — discarding this late capture.")
                            continue
                        opening_turn_claimed["value"] = True
                       
                        detector.silence_frames_needed = _normal_silence_frames_needed
                    duration_s = len(turn_audio) / 2 / call_sample_rate
                    padding_s = (VAD_SETTINGS["vad_preroll_ms"] + VAD_SETTINGS["vad_silence_ms"]) / 1000
                    estimated_speech_s = max(0.0, duration_s - padding_s)
                    
                    turn_rms = _compute_rms(turn_audio)

                    resampled = resample_pcm16(turn_audio, call_sample_rate, WHISPER_SAMPLE_RATE)
                    whisper_input = pcm16_to_whisper_input(resampled)
                    raw_wav_path = _save_debug_wav(turn_audio, call_sample_rate, "raw8k")
                    whisper_wav_path = _save_debug_wav(resampled.tobytes(), WHISPER_SAMPLE_RATE, "whisper16k")

                    result = await stt.transcribe_array(
                        whisper_input, estimated_speech_s,
                        prompt=NAME_HINT_PROMPT if chat.waiting_for_name else None,
                    )
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
                    print(f"[DEBUG-AUDIO] raw={raw_wav_path.name} whisper_input={whisper_wav_path.name}")

                   
                    bare_check = result["text"].strip().lower().rstrip(".,!?")
                    if chat.call_finished:
                     
                        print(f"[FILTER] Discarding turn — call already finished: {result['text']!r}")
                    elif bare_check in GRATITUDE_PHRASES:
                        print(f"[FILTER] Discarding likely-hallucinated {result['text']!r} (known Whisper noise artifact)")
                    elif logprob is not None and logprob < CONFIDENCE_REJECT_THRESHOLD:
                      
                        print(f"[FILTER] Low confidence (avg_logprob={logprob:.3f}) — asking customer to repeat")
                        reply_text = chat.respond_low_confidence()
                        print(f"[SPARK] {reply_text}")
                        pcm_bytes, native_rate = tts.synthesize_to_pcm(reply_text)
                        pcm_8k = resample_pcm16(pcm_bytes, native_rate, call_sample_rate).tobytes()
                        await speak(ws, stream_sid, pcm_8k, call_sample_rate, spark_speaking, spark_state=spark_state, text=reply_text)
                        last_activity_at = time.time()
                        silence_prompted = False
                    elif chat.waiting_for_opening_pitch or result["text"].strip():
                       
                        last_activity_at = time.time()
                        silence_prompted = False

                       
                        turn_text = result["text"].strip() or (
                            NO_RESPONSE_SENTINEL if chat.waiting_for_opening_pitch else ""
                        )
                        print(f"[TIMING-DEBUG] run_turn() starting — wall clock: {time.strftime('%H:%M:%S.')}{int(time.time()*1000)%1000:03d}")

                        
                        previous_speech_task = speech_task
                        if pending_interruption_check_task is not None and not pending_interruption_check_task.done():
                           
                            pending_interruption_check_task.cancel()
                        speech_task = asyncio.create_task(start_new_turn(turn_text, previous_speech_task))

            elif event_type == "stop":
                print("[EXOTEL] stop — call ended.")
                if speech_task is not None and not speech_task.done():
                    speech_task.cancel()
                break

    except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError, asyncio.TimeoutError) as e:
        
        print(f"[EXOTEL] Connection error, ending this call: {e}")
        call_errored = True

    finally:

        call_ended.set()

        if speech_task is not None and not speech_task.done():
            speech_task.cancel()

        if chat is not None:
            # simulate_exotel_call.py 
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

           
            needs_restart = _completed_call_count >= MAX_CALLS_BEFORE_RESTART

            if needs_restart:
                reason = f"reached the {MAX_CALLS_BEFORE_RESTART}-call safety-net limit"
                print(
                    f"[RESTART] Restarting the process ({reason}) — releasing "
                    f"accumulated memory before it affects the next call's "
                    f"latency. The launcher script should relaunch "
                    f"automatically. Skipping this cycle's dial — the "
                    f"restarted process will resume the batch on startup."
                )
                _shutdown_event.set()
            elif BATCH_ACTIVE_PATH.exists():
                
                await _continue_batch()


async def main():
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    print("Two-way pipeline test server listening on ws://0.0.0.0:5001")
    print("Run 'ngrok http 5001' (or cloudflared) in another terminal to expose this publicly.")
    async with websockets.serve(handle, "0.0.0.0", 5001):
        
        if BATCH_ACTIVE_PATH.exists():
            print("[RESTART] A batch was active before this restart — resuming it now.")
            await _continue_batch()

        await _shutdown_event.wait()
    print("[RESTART] Server closed. Exiting process for the launcher to relaunch.")


if __name__ == "__main__":
    asyncio.run(main())