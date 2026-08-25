"""
Turn detection + real STT test server — milestone 2 of the real
telephony pipeline. Builds directly on the proven turn detector; the
only new piece here is feeding each detected turn's audio into the
SAME Whisper model the local-mic version already uses (loaded once at
startup, exactly like VoiceSession does), instead of just printing a
byte count.

Doesn't reply to anything yet — chat_session isn't wired in at this
stage. Just proves turn-timing AND transcription quality together,
which byte counts alone couldn't show (e.g. a pause cutting off the
end of a word wouldn't show up as a "bad" byte count, but WOULD show
up as visibly wrong transcribed text).

Run from the PROJECT ROOT (not from inside telephony/), so app.stt.whisper
is importable:
    python -m telephony.stream_stt_test

Setup (same as the turn-detection-only test, nothing new):
    pip install websockets webrtcvad-wheels --break-system-packages
"""
import asyncio
import json
import time

import websockets

from telephony.stream_vad import CallTurnDetector
from telephony.audio_utils import resample_pcm16, pcm16_to_whisper_input
from telephony.dual_tier_stt import DualTierSTT

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

# Loaded once, here, at import time — not per-call, not per-turn. Same
# "load the expensive thing once, reuse it" principle as VoiceSession.
# Loads BOTH models upfront (more RAM, one-time longer startup) so
# there's no reload cost mid-call when a short turn needs the more
# accurate one.
stt = DualTierSTT(fast_model="base", accurate_model="small")


async def handle(ws):
    detector = None
    call_sample_rate = 8000
    print("[EXOTEL] Connection opened.")

    async for raw_message in ws:
        event = json.loads(raw_message)
        event_type = event.get("event")

        if event_type == "connected":
            print("[EXOTEL] connected")

        elif event_type == "start":
            media_format = event.get("start", {}).get("media_format", {})
            call_sample_rate = int(media_format.get("sample_rate", 8000))
            print(f"[EXOTEL] start — media_format={media_format}")
            detector = CallTurnDetector(sample_rate=call_sample_rate, settings=VAD_SETTINGS)
            print(f"[EXOTEL] Turn detector ready at {call_sample_rate}Hz — listening...")

        elif event_type == "media":
            if detector is None:
                continue
            payload = event["media"]["payload"]
            turn_audio = detector.add_media_payload(payload)

            if turn_audio is not None:
                duration_s = len(turn_audio) / 2 / call_sample_rate
                timestamp = time.strftime("%H:%M:%S")
                print(f"[TURN DETECTED @ {timestamp}] ~{duration_s:.2f}s — transcribing...")

                # The reported duration includes the VAD preroll buffer
                # AND the full trailing silence needed to detect the
                # turn ended — roughly (vad_preroll_ms + vad_silence_ms)
                # of non-speech padding on every single turn, regardless
                # of how short the actual word was. Comparing the RAW
                # duration against the short-turn threshold meant it
                # never triggered — even a brief single word measured
                # well above 2s once padding was included. Subtracting
                # the known padding gives a much closer estimate of
                # actual speech length to route on.
                padding_s = (VAD_SETTINGS["vad_preroll_ms"] + VAD_SETTINGS["vad_silence_ms"]) / 1000
                estimated_speech_s = max(0.0, duration_s - padding_s)

                resampled = resample_pcm16(turn_audio, call_sample_rate, WHISPER_SAMPLE_RATE)
                whisper_input = pcm16_to_whisper_input(resampled)

                result = stt.transcribe_array(whisper_input, estimated_speech_s)
                confidence = result["avg_logprob"]
                confidence_str = f"{confidence:.3f}" if confidence is not None else "N/A"
                print(f"[TRANSCRIPT] \"{result['text']}\" (model={result['model_used']}, est_speech={estimated_speech_s:.2f}s, avg_logprob={confidence_str}, took {result['duration']:.2f}s)")

        elif event_type == "stop":
            print("[EXOTEL] stop — call ended.")
            break


async def main():
    print("Turn detection + STT test server listening on ws://0.0.0.0:5001")
    print("Run 'ngrok http 5001' (or cloudflared) in another terminal to expose this publicly.")
    async with websockets.serve(handle, "0.0.0.0", 5001):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())