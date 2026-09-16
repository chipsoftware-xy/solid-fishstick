"""
test_vad_stt_pipeline.py

Standalone test harness for the updated VAD trimming (stream_vad.py's
CallTurnDetector) + the updated DeepInfra STT settings
(deepinfra_stt.py: response_format="verbose_json", temperature=0.0,
service_tier=None by default).

Feeds a WAV file through CallTurnDetector exactly the way Exotel's real
`media` events would — base64-encoded PCM16 chunks, pushed sequentially,
paced in real time by default — and for every detected turn:

  - measures how long add_media_payload() itself takes (VAD compute
    cost — should be near-zero; confirms VAD isn't the bottleneck)
  - measures detection latency (wall-clock time from a turn's first
    speech frame to the moment CallTurnDetector returns it — should
    roughly track vad_silence_ms, since that's the dominant cost)
  - runs the turn through the real STT class and logs stt_time,
    avg_logprob, no_speech_prob, compression_ratio, and the transcript

Place this in your project root (same level as two_way_test.py) so the
telephony.* / app.* imports resolve.

Usage:
    python test_vad_stt_pipeline.py path/to/test_call.wav
    python test_vad_stt_pipeline.py path/to/test_call.wav --fast
    python test_vad_stt_pipeline.py path/to/test_call.wav --save-debug-audio
"""
import argparse
import asyncio
import base64
import time
import wave
from pathlib import Path

import numpy as np

from telephony.stream_vad import CallTurnDetector, _compute_rms
from telephony.audio_utils import resample_pcm16, pcm16_to_whisper_input
from app.stt.deepinfra_stt import STT

EXOTEL_SAMPLE_RATE = 8000
WHISPER_SAMPLE_RATE = 16000

# Exotel sends ~20ms media chunks in real calls — chunk the test file the
# same way so add_media_payload()'s leftover-buffering logic gets
# exercised the same as it is in production, rather than fed in neat
# VAD-frame-sized (30ms) pieces which would skip that code path.
CHUNK_MS = 20

# Same settings as two_way_test.py's VAD_SETTINGS — keep these in sync
# manually, or import VAD_SETTINGS from two_way_test.py directly if you'd
# rather guarantee they never drift apart.
VAD_SETTINGS = {
    "vad_aggressiveness": 2,
    "vad_frame_ms": 30,
    "vad_preroll_ms": 300,
    "vad_silence_ms": 800,
    "vad_min_speech_ms": 300,
    "vad_max_utterance_ms": 20000,
    "vad_min_rms": 700,
}


def load_wav_as_pcm16(path, target_rate=EXOTEL_SAMPLE_RATE):
    """
    Loads a WAV file and returns mono PCM16 bytes at target_rate.
    Downmixes stereo (left channel only) and resamples if needed, so any
    WAV can be dropped in without pre-processing it by hand first.
    """
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM, got {sample_width * 8}-bit — "
            f"convert the file to 16-bit PCM WAV first."
        )

    if n_channels == 2:
        samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        raw = samples[:, 0].tobytes()
    elif n_channels != 1:
        raise ValueError(f"Expected mono or stereo, got {n_channels} channels.")

    if frame_rate != target_rate:
        raw = resample_pcm16(raw, frame_rate, target_rate).tobytes()

    return raw


async def run_test(wav_path, save_debug_audio=False, realtime=True):
    print(f"Loading {wav_path}...")
    pcm_bytes = load_wav_as_pcm16(wav_path)
    total_duration_s = len(pcm_bytes) / 2 / EXOTEL_SAMPLE_RATE
    print(f"Loaded {total_duration_s:.1f}s of audio @ {EXOTEL_SAMPLE_RATE}Hz")
    print(f"Pacing: {'real-time (matches a live call)' if realtime else 'as fast as possible (--fast)'}")
    if not realtime:
        print("NOTE: detection_latency numbers below are meaningless without real-time pacing — "
              "they'll show near-zero regardless of vad_silence_ms. Drop --fast to get real numbers.")
    print()

    detector = CallTurnDetector(sample_rate=EXOTEL_SAMPLE_RATE, settings=VAD_SETTINGS)
    stt = STT()
    print("Verifying DeepInfra STT access...")
    await stt.verify()
    print()

    chunk_bytes = int(EXOTEL_SAMPLE_RATE * CHUNK_MS / 1000) * 2  # 16-bit = 2 bytes/sample
    chunk_duration_s = CHUNK_MS / 1000

    debug_dir = Path("test_debug_audio")
    if save_debug_audio:
        debug_dir.mkdir(exist_ok=True)

    turn_index = 0
    vad_processing_total = 0.0
    turn_start_wall_time = None
    results = []

    for i in range(0, len(pcm_bytes), chunk_bytes):
        chunk = pcm_bytes[i:i + chunk_bytes]
        if not chunk:
            continue
        payload = base64.b64encode(chunk).decode()

        was_triggered_before = detector.triggered

        _vad_start = time.perf_counter()
        turn_audio = detector.add_media_payload(payload)
        vad_processing_total += time.perf_counter() - _vad_start

        if not was_triggered_before and detector.triggered:
            # This chunk is what flipped the detector into "collecting
            # a turn" state — start the clock here.
            turn_start_wall_time = time.perf_counter()

        if turn_audio is not None:
            detection_latency = (
                time.perf_counter() - turn_start_wall_time
                if turn_start_wall_time is not None else None
            )
            turn_start_wall_time = None

            turn_index += 1
            turn_duration_s = len(turn_audio) / 2 / EXOTEL_SAMPLE_RATE
            turn_rms = _compute_rms(turn_audio)

            if save_debug_audio:
                debug_path = debug_dir / f"turn_{turn_index:03d}.wav"
                with wave.open(str(debug_path), "wb") as out_wf:
                    out_wf.setnchannels(1)
                    out_wf.setsampwidth(2)
                    out_wf.setframerate(EXOTEL_SAMPLE_RATE)
                    out_wf.writeframes(turn_audio)

            resampled = resample_pcm16(turn_audio, EXOTEL_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
            whisper_input = pcm16_to_whisper_input(resampled)

            _stt_start = time.perf_counter()
            result = await stt.transcribe_array(whisper_input, turn_duration_s)
            stt_wall_time = time.perf_counter() - _stt_start

            logprob = result.get("avg_logprob")
            no_speech = result.get("no_speech_prob")
            compression = result.get("compression_ratio")
            latency_str = f"{detection_latency:.3f}s" if detection_latency is not None else "N/A"

            print(
                f"[Turn {turn_index}] duration={turn_duration_s:.2f}s rms={turn_rms:.1f} "
                f"detection_latency={latency_str} "
                f"stt_time={stt_wall_time:.2f}s (reported={result['duration']:.2f}s)\n"
                f"    avg_logprob={logprob} no_speech_prob={no_speech} compression_ratio={compression}\n"
                f"    text={result['text']!r}\n"
            )

            results.append({
                "turn_index": turn_index,
                "duration_s": turn_duration_s,
                "rms": turn_rms,
                "detection_latency_s": detection_latency,
                "stt_time_s": stt_wall_time,
                "avg_logprob": logprob,
                "no_speech_prob": no_speech,
                "compression_ratio": compression,
                "text": result["text"],
            })

        if realtime:
            await asyncio.sleep(chunk_duration_s)

    print(f"{'=' * 70}")
    print(f"Total audio: {total_duration_s:.1f}s")
    print(f"Turns detected: {turn_index}")
    print(f"Total VAD compute time (sum of all add_media_payload() calls): "
          f"{vad_processing_total * 1000:.1f}ms")

    if results:
        stt_times = [r["stt_time_s"] for r in results]
        print(f"\nSTT time — mean: {sum(stt_times) / len(stt_times):.2f}s, "
              f"median: {sorted(stt_times)[len(stt_times) // 2]:.2f}s, "
              f"min: {min(stt_times):.2f}s, max: {max(stt_times):.2f}s")

        det_latencies = [r["detection_latency_s"] for r in results if r["detection_latency_s"] is not None]
        if det_latencies:
            print(f"Detection latency — mean: {sum(det_latencies) / len(det_latencies):.3f}s, "
                  f"min: {min(det_latencies):.3f}s, max: {max(det_latencies):.3f}s "
                  f"(compare against vad_silence_ms={VAD_SETTINGS['vad_silence_ms']}ms)")

        slow_turns = [r for r in results if r["stt_time_s"] > 5.0]
        if slow_turns:
            print(f"\nSlow turns (stt_time > 5s) — check these for retry-storm/hallucination signs:")
            for r in slow_turns:
                print(f"  Turn {r['turn_index']}: stt_time={r['stt_time_s']:.2f}s, "
                      f"compression_ratio={r['compression_ratio']}, "
                      f"no_speech_prob={r['no_speech_prob']}, "
                      f"avg_logprob={r['avg_logprob']}, text={r['text']!r}")
        else:
            print("\nNo turns exceeded 5s STT time.")
    else:
        print("\nNo turns were detected at all — check vad_min_rms / vad_aggressiveness "
              "against this file's actual levels, or confirm the file has real speech in it.")


def main():
    parser = argparse.ArgumentParser(
        description="Test the CallTurnDetector VAD trim + DeepInfra STT pipeline against a WAV file."
    )
    parser.add_argument("wav_path", type=str, help="Path to the test audio WAV file")
    parser.add_argument(
        "--fast", action="store_true",
        help="Process as fast as possible instead of real-time pacing. "
             "Faster iteration, but detection_latency numbers become meaningless."
    )
    parser.add_argument(
        "--save-debug-audio", action="store_true",
        help="Save each detected (and trimmed) turn as a WAV file in test_debug_audio/"
    )
    args = parser.parse_args()

    asyncio.run(run_test(
        Path(args.wav_path),
        save_debug_audio=args.save_debug_audio,
        realtime=not args.fast,
    ))


if __name__ == "__main__":
    main()