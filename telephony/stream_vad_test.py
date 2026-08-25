"""
Turn-detection test server — milestone 1 of the real telephony
pipeline. Doesn't transcribe or reply to anything yet; just proves the
buffering/VAD layer correctly identifies "the customer stopped talking"
at the right moments against REAL call audio, before wiring in STT.

Place a real call (same Flow/Voicebot setup already proven working),
talk, pause, talk again, and confirm "[TURN DETECTED]" prints roughly
when you actually paused — not way too early, not way too late.

Setup:
    pip install websockets webrtcvad-wheels --break-system-packages

Run:
    python stream_vad_test.py

Then expose it the same way as before (ngrok/cloudflared), and place a
test call through your existing Flow with this new server's wss:// URL
configured in the Voicebot applet.
"""
import asyncio
import base64
import json
import time

import websockets
from stream_vad import CallTurnDetector

# Same defaults as config/settings.yml's audio section — these were
# tuned against local mic input, not phone audio, so expect to re-tune
# once you hear how well turn boundaries actually line up on a real call.
VAD_SETTINGS = {
    "vad_aggressiveness": 2,
    "vad_frame_ms": 30,
    "vad_preroll_ms": 300,
    "vad_silence_ms": 800,
    "vad_min_speech_ms": 300,
    "vad_max_utterance_ms": 20000,
}


async def handle(ws):
    detector = None
    sample_rate = 8000  # overwritten below once the real start event tells us
    print("[EXOTEL] Connection opened.")

    async for raw_message in ws:
        event = json.loads(raw_message)
        event_type = event.get("event")

        if event_type == "connected":
            print("[EXOTEL] connected")

        elif event_type == "start":
            media_format = event.get("start", {}).get("media_format", {})
            sample_rate = int(media_format.get("sample_rate", 8000))
            print(f"[EXOTEL] start — media_format={media_format}")
            detector = CallTurnDetector(sample_rate=sample_rate, settings=VAD_SETTINGS)
            print(f"[EXOTEL] Turn detector ready at {sample_rate}Hz — listening...")

        elif event_type == "media":
            if detector is None:
                continue  # media arrived before start somehow — skip rather than crash
            payload = event["media"]["payload"]
            turn_audio = detector.add_media_payload(payload)
            if turn_audio is not None:
                duration_s = len(turn_audio) / 2 / sample_rate
                timestamp = time.strftime("%H:%M:%S")
                print(f"[TURN DETECTED @ {timestamp}] {len(turn_audio)} bytes (~{duration_s:.2f}s)")

        elif event_type == "stop":
            print("[EXOTEL] stop — call ended.")
            break


async def main():
    print("Turn-detection test server listening on ws://0.0.0.0:5001")
    print("Run 'ngrok http 5001' (or cloudflared) in another terminal to expose this publicly.")
    async with websockets.serve(handle, "0.0.0.0", 5001):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())