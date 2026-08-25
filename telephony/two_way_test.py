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

import websockets

from telephony.stream_vad import CallTurnDetector
from telephony.audio_utils import resample_pcm16, pcm16_to_whisper_input
from telephony.dual_tier_stt import DualTierSTT
from app.tts.piper import TTS
from app.llm.chat_session import ChatSession
from app.llm.ollama_client import LLM
from app.knowledge_base import KnowledgeBase
from app.call_finalizer import finalize_call

VAD_SETTINGS = {
    "vad_aggressiveness": 2,
    "vad_frame_ms": 30,
    "vad_preroll_ms": 300,
    "vad_silence_ms": 800,
    "vad_min_speech_ms": 300,
    "vad_max_utterance_ms": 20000,
    "vad_min_rms": 150,
}

WHISPER_SAMPLE_RATE = 16000
EXOTEL_SAMPLE_RATE = 8000
# Within Exotel's documented 3200-100000 byte range, a multiple of 320.
# 3200 bytes at 8kHz/16-bit mono = 200ms of audio per outgoing chunk.
OUTGOING_CHUNK_BYTES = 3200

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


async def handle(ws):
    detector = None
    chat = None
    stream_sid = None
    customer_phone = None
    call_sample_rate = 8000
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
                customer_phone = start_data.get("from")
                media_format = start_data.get("media_format", {})
                call_sample_rate = int(media_format.get("sample_rate", 8000))
                print(f"[EXOTEL] start — media_format={media_format}")

                detector = CallTurnDetector(sample_rate=call_sample_rate, settings=VAD_SETTINGS)
                chat = ChatSession(llm=llm, knowledge=knowledge)

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

            elif event_type == "media":
                if detector is None:
                    continue
                payload = event["media"]["payload"]
                turn_audio = detector.add_media_payload(payload)

                if turn_audio is not None:
                    duration_s = len(turn_audio) / 2 / call_sample_rate
                    padding_s = (VAD_SETTINGS["vad_preroll_ms"] + VAD_SETTINGS["vad_silence_ms"]) / 1000
                    estimated_speech_s = max(0.0, duration_s - padding_s)

                    resampled = resample_pcm16(turn_audio, call_sample_rate, WHISPER_SAMPLE_RATE)
                    whisper_input = pcm16_to_whisper_input(resampled)

                    result = stt.transcribe_array(whisper_input, estimated_speech_s)
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[CUSTOMER @ {timestamp}] {result['text']!r} (model={result['model_used']})")

                    if result["text"].strip():
                        await run_turn(ws, chat, stream_sid, result["text"])

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


async def main():
    print("Two-way pipeline test server listening on ws://0.0.0.0:5001")
    print("Run 'ngrok http 5001' (or cloudflared) in another terminal to expose this publicly.")
    async with websockets.serve(handle, "0.0.0.0", 5001):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())