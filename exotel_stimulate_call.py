"""
Simulates a real Exotel call against your ACTUAL two_way_test.py
server — not a mock, not a separate test path. This connects over a
real WebSocket and speaks Exotel's exact message protocol (connected/
start/media/stop events, base64 PCM16 audio), so every piece of logic
that only exists in the telephony path runs for real: VAD/turn
detection, the stall filler, the proactive silence check-in (10s/20s
timers), STT confidence scoring, and the mute-window/echo-protection
logic. None of that runs during a local-mic (voice_session.py) test,
since that's a separate code path entirely.

What this CAN'T validate: real PSTN network jitter, actual Exotel
infrastructure quirks, or ngrok tunnel behavior under Exotel's real
traffic. Those genuinely need a small number of real calls — this is
for catching everything else, for free, as many times as you want.

SETUP:
    1. Make sure telephony/two_way_test.py is already running:
       python -m telephony.two_way_test
    2. Record a few WAV clips of yourself saying different things -
       reuse your mic, same as local-mic testing. Realistic phone
       conversation turns, e.g.:
         turn1.wav: "Yeah sure, go ahead"
         turn2.wav: "We need about 30 PVC cards for our team"
         turn3.wav: "Sure, connect me with someone"
    3. (Recommended) Degrade each clip to phone-quality using the
       existing simulate_phone_audio.py first, so STT confidence
       scoring sees realistic audio, not clean mic input:
         python simulate_phone_audio.py turn1.wav turn1_phone.wav
    4. Edit CALL_SCRIPT below to point at your actual clip paths.

USAGE:
    python simulate_exotel_call.py                  # runs CALL_SCRIPT
    python simulate_exotel_call.py --silence-test    # tests the 10s/20s
                                                      # silence check-in
                                                      # instead, by
                                                      # sending near-
                                                      # silent audio and
                                                      # doing nothing
"""
import asyncio
import base64
import json
import sys
import time
import wave

import numpy as np
import websockets

SERVER_URL = "ws://localhost:5001"
EXOTEL_SAMPLE_RATE = 8000
CHUNK_BYTES = 3200  # matches OUTGOING_CHUNK_BYTES in two_way_test.py - 200ms per chunk at 8kHz/16-bit
FAKE_PHONE_NUMBER = "09999999999"  # domestic format, like Exotel actually sends

# Edit these to your actual recorded clip paths, in the order you want
# them "spoken" during the simulated call. Silence between entries is
# handled automatically (waits for Spark's reply to finish before
# sending the next one).
CALL_SCRIPT = [
    "audio/input/test2.wav",
    "audio/input/test3.wav",
    "audio/input/test4.wav",
    "audio/input/test5.wav",
    "audio/input/test6.wav",
    "audio/input/test7.wav",
    "audio/input/test8-0.wav",
    "audio/input/test8-1.wav",
    "audio/input/test9.wav",
    "audio/input/test10.wav",
    "audio/input/test11.wav",
]


def load_wav_as_pcm16_8k(path):
    """Loads a WAV file and returns raw PCM16 bytes at 8kHz mono.

    Uses scipy's resample_poly for downsampling (proper anti-aliasing
    filter included) rather than naive linear interpolation. This
    matters specifically here: in a REAL Exotel call, audio arrives
    natively at 8kHz with no resampling step at all. This script's
    resampling step only exists because your recorded clips are
    likely 44100Hz/48000Hz - a resampling artifact naive linear
    interpolation can introduce is exactly the kind of thing that
    could make webrtcVAD's speech classifier never recognize it as
    speech, causing a silent, total VAD failure with no error at all -
    a failure mode that wouldn't exist in a real call, so worth ruling
    out properly rather than leaving it as a guess.
    """
    import scipy.signal

    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(f"{path}: expected 16-bit audio, got {sample_width * 8}-bit")

    samples = np.frombuffer(frames, dtype=np.int16)
    if n_channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

    original_rate = sample_rate
    if sample_rate != EXOTEL_SAMPLE_RATE:
        # resample_poly needs integer up/down ratios - reduce the
        # fraction first rather than pass huge numbers directly.
        from math import gcd
        g = gcd(EXOTEL_SAMPLE_RATE, sample_rate)
        up = EXOTEL_SAMPLE_RATE // g
        down = sample_rate // g
        samples = scipy.signal.resample_poly(samples, up, down).astype(np.int16)

    # Trim leading silence - a recorded clip's own lead-in (e.g. ~2s
    # before you start talking) otherwise eats directly into the
    # server's 10-second silence-check-in budget for no benefit,
    # since a real, prompt response wouldn't have that dead air.
    # Simple RMS-window scan rather than anything fancier.
    window_samples = int(0.05 * EXOTEL_SAMPLE_RATE)  # 50ms windows
    trim_start = 0
    for i in range(0, len(samples) - window_samples, window_samples):
        window = samples[i:i + window_samples]
        window_rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
        if window_rms > 700:  # matches the server's own vad_min_rms floor
            trim_start = max(0, i - window_samples)  # keep a small pre-roll
            break
    if trim_start > 0:
        trimmed_seconds = trim_start / EXOTEL_SAMPLE_RATE
        print(f"  Trimmed {trimmed_seconds:.2f}s of leading silence")
        samples = samples[trim_start:]

    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.0
    duration = len(samples) / EXOTEL_SAMPLE_RATE
    print(
        f"  Loaded: {original_rate}Hz -> {EXOTEL_SAMPLE_RATE}Hz, "
        f"{duration:.2f}s, RMS={rms:.1f} "
        f"(server's vad_min_rms floor is 700 - well below that means "
        f"this clip likely won't register as speech at all)"
    )

    return samples.tobytes()


async def send_audio_as_media_events(ws, stream_sid, pcm_bytes, chunk_delay=0.2):
    """Sends PCM audio as a series of media events, paced in real time
    (not all at once) - the VAD and silence-detection logic depend on
    real wall-clock timing between chunks, same as a real call."""
    for i in range(0, len(pcm_bytes), CHUNK_BYTES):
        chunk = pcm_bytes[i:i + CHUNK_BYTES]
        payload = base64.b64encode(chunk).decode()
        await ws.send(json.dumps({
            "event": "media",
            "stream_sid": stream_sid,
            "media": {"payload": payload},
        }))
        await asyncio.sleep(chunk_delay)


async def wait_for_spark_to_finish_speaking(ws, quiet_period=1.5, max_wait=30.0, safety_margin=2.0):
    """Listens for incoming media events (Spark's TTS audio) and
    returns once no new one has arrived for `quiet_period` seconds,
    then waits an additional safety_margin seconds to clear the
    server's own mute_until window (1.0s + reply_duration * 0.15).

    safety_margin was previously 4.0 - more than double what even the
    longest reply (the opening message) actually needs (~2.8-3.25s).
    That excess delay, combined with a clip's own leading silence, was
    eating directly into the 10-second silence-check-in budget before
    any real speech could register - confirmed directly: it caused
    the check-in prompt to fire mid-transmission of a real clip, which
    then set a NEW mute window that chopped the still-sending audio
    off mid-stream, producing a mangled partial transcript ("Yai!"
    instead of "yeah sure") rather than a clean one.

    Returns the total bytes of audio actually received during this
    wait - lets the caller distinguish a real, long reply (like the
    opening message) from a suspiciously short one (like the "we're
    already on a call" busy-rejection message), which matters because
    proceeding to send scripted audio into a connection that's about
    to be closed produces a confusing, misleading error otherwise."""
    start = time.time()
    last_received = time.time()
    total_bytes_received = 0
    while True:
        remaining = quiet_period - (time.time() - last_received)
        if remaining <= 0:
            break
        if time.time() - start > max_wait:
            print("  (max wait exceeded - continuing anyway)")
            break
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            event = json.loads(raw)
            if event.get("event") == "media":
                last_received = time.time()
                payload = event.get("media", {}).get("payload", "")
                total_bytes_received += len(base64.b64decode(payload)) if payload else 0
        except asyncio.TimeoutError:
            break

    await asyncio.sleep(safety_margin)
    return total_bytes_received


async def send_silence(ws, stream_sid, seconds):
    """Sends TRUE silence (zero-amplitude samples) for the given
    duration. Previously used low-amplitude random noise, which
    turned out to be the actual bug: random broadband noise, even
    quiet, has spectral characteristics that can fool webrtcVAD's
    classifier into occasionally calling it speech - confirmed
    directly via debug logging (a frame at RMS=160.6, clearly quiet,
    still classified as is_speech=True). That kept silence_run from
    ever reaching the ~27 consecutive non-speech frames needed to
    close out a turn, so utterances never completed even though the
    VAD was correctly detecting the real speech portions. True zero
    silence has no spectral content at all, so it's unambiguous."""
    samples_needed = int(seconds * EXOTEL_SAMPLE_RATE)
    silence = np.zeros(samples_needed, dtype=np.int16)
    pcm_bytes = silence.tobytes()
    await send_audio_as_media_events(ws, stream_sid, pcm_bytes, chunk_delay=0.2)


async def run_call_script(clip_paths):
    print(f"Connecting to {SERVER_URL} ...")
    async with websockets.connect(SERVER_URL) as ws:
        await ws.send(json.dumps({"event": "connected"}))

        stream_sid = "SIMULATED_" + str(int(time.time()))
        await ws.send(json.dumps({
            "event": "start",
            "start": {
                "stream_sid": stream_sid,
                "from": FAKE_PHONE_NUMBER,
                "media_format": {
                    "encoding": "base64",
                    "sample_rate": str(EXOTEL_SAMPLE_RATE),
                    "bit_rate": "128kbps",
                },
            },
        }))

        print(f"[{time.strftime('%H:%M:%S')}] Waiting for Spark's opening message...")
        opening_bytes = await wait_for_spark_to_finish_speaking(ws)

        # The real opening message is long (~10s+ of audio, several
        # tens of KB at 8kHz/16-bit). The busy-rejection message
        # ("We will contact you soon...") is much shorter. This
        # threshold is comfortably below the real opening's size but
        # above the busy message's, so a short reply here reliably
        # means this connection was declined, not accepted.
        if opening_bytes < 40000:
            print(
                f"  Received only {opening_bytes} bytes for the opening reply — "
                f"this looks like the busy-rejection message, not the real "
                f"opening. The server is likely already on another call. "
                f"Stopping here rather than sending scripted turns into a "
                f"connection that's about to close."
            )
            return

        for i, clip_path in enumerate(clip_paths, 1):
            ts = time.strftime("%H:%M:%S")
            print(f"\n[{ts}] --- Turn {i}: sending {clip_path} ---")
            try:
                pcm_bytes = load_wav_as_pcm16_8k(clip_path)
            except FileNotFoundError:
                print(f"  ERROR: {clip_path} not found - check CALL_SCRIPT paths. Skipping.")
                continue

            try:
                await send_audio_as_media_events(ws, stream_sid, pcm_bytes)
                # Send trailing silence — a real, continuously-open phone
                # line naturally provides ambient silence after the
                # customer stops talking, which is what the VAD's
                # ~800ms silence-after-speech window needs to declare
                # the utterance complete. This script otherwise just
                # stops sending anything after the clip ends, which
                # left the VAD "triggered" indefinitely with nothing
                # to complete the turn — confirmed as the likely cause
                # of turns never completing despite healthy RMS.
                await send_silence(ws, stream_sid, seconds=1.5)
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}]   Sent (plus 1.5s trailing silence). Waiting for Spark's reply...")
                await wait_for_spark_to_finish_speaking(ws)
            except websockets.exceptions.ConnectionClosed:
                print(
                    "  Connection closed by the server before this turn "
                    "completed - likely the silence-hangup logic fired "
                    "because a PREVIOUS turn's audio didn't register as "
                    "speech (arrived too soon after the prior reply, "
                    "check the server's own logs for what it actually "
                    "heard). Stopping here rather than continuing against "
                    "a dead connection."
                )
                return

        print("\n--- Sending stop event ---")
        try:
            await ws.send(json.dumps({"event": "stop"}))
            await asyncio.sleep(1)
        except websockets.exceptions.ConnectionClosed:
            print("  (connection was already closed - nothing to send)")

    print("\nSimulated call finished - check the two_way_test.py terminal for full logs.")


async def run_silence_test():
    """Sends one real turn, then goes silent for 25 seconds - long
    enough to see both the 10s check-in prompt AND the 20s hangup
    fire, back to back, in the two_way_test.py terminal."""
    print(f"Connecting to {SERVER_URL} ...")
    async with websockets.connect(SERVER_URL) as ws:
        await ws.send(json.dumps({"event": "connected"}))
        stream_sid = "SIMULATED_SILENCE_" + str(int(time.time()))
        await ws.send(json.dumps({
            "event": "start",
            "start": {
                "stream_sid": stream_sid,
                "from": FAKE_PHONE_NUMBER,
                "media_format": {
                    "encoding": "base64",
                    "sample_rate": str(EXOTEL_SAMPLE_RATE),
                    "bit_rate": "128kbps",
                },
            },
        }))

        print(f"[{time.strftime('%H:%M:%S')}] Waiting for Spark's opening message...")
        await wait_for_spark_to_finish_speaking(ws)

        print("\nGoing silent for 25 seconds - watch the two_way_test.py")
        print("terminal for the check-in prompt (~10s) and hangup (~20s)...")
        await send_silence(ws, stream_sid, seconds=25)
        await wait_for_spark_to_finish_speaking(ws, max_wait=15)

    print("\nSilence test finished - check the two_way_test.py terminal.")


if __name__ == "__main__":
    if "--silence-test" in sys.argv:
        asyncio.run(run_silence_test())
    else:
        asyncio.run(run_call_script(CALL_SCRIPT))