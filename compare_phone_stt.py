"""
Direct test of the specific mechanism suspected earlier: does Whisper
genuinely struggle with a short, isolated utterance (like a bare
"Hello") preceded by real silence — an acoustic/recognition problem,
not a connection-warmth problem — using a real clip that actually
contains that exact shape (a few seconds of genuine silence, then
"Hello"), rather than inferring "short utterance" from how a fixed-
length slice happened to land.

Tests TWO variants of the same clip:
  1. FULL — the clip exactly as recorded, silence and all, sent to
     Whisper in one piece.
  2. VAD-TRIMMED — the clip run through the same real CallTurnDetector
     production actually uses, extracting only what the real pipeline
     would actually capture and send (a short preroll plus the speech
     itself, NOT the full leading silence) — run at 8kHz first, same as
     Exotel's native rate, then resampled to 16kHz for Whisper, exactly
     matching the real two-stage pipeline.

This directly answers two separate questions at once:
  - Does a bare "Hello" transcribe slowly/incorrectly regardless of how
    much leading silence is included? (An acoustic problem with the
    utterance itself.)
  - Is our own VAD trimming making this better or worse — is it cutting
    away something Whisper actually needed, or successfully stripping
    away exactly the part that doesn't matter?

SETUP: place the file at audio/input/test1.wav (or edit CLIP_PATH below)
— same DeepInfra key requirement as the other scripts in this set.

USAGE:
    python test_isolated_hello.py
"""
import asyncio
import base64
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from test import load_and_resample
from app.stt.deepinfra_stt import STT as DeepInfraSTT
from telephony.stream_vad import CallTurnDetector

CLIP_PATH = "audio/input/trimtest.wav"
REPEATS = 5  # Per variant — averages out the substantial per-call
             # variance already confirmed in the earlier tests.

VAD_SAMPLE_RATE = 8000    # Matches Exotel's native rate, same as production.
WHISPER_SAMPLE_RATE = 16000  # Matches what Whisper/DeepInfra expects.

VAD_SETTINGS = {
    "vad_aggressiveness": 2,
    "vad_frame_ms": 30,
    "vad_preroll_ms": 300,
    "vad_silence_ms": 1200,
    "vad_min_speech_ms": 300,
    "vad_max_utterance_ms": 20000,
    "vad_min_rms": 700,
}


def float32_to_pcm16_bytes(samples):
    return np.clip(samples * 32768.0, -32768, 32767).astype(np.int16).tobytes()


def pcm16_bytes_to_float32(raw_bytes):
    return (np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0)


def resample_float32(samples, from_rate, to_rate):
    if from_rate == to_rate:
        return samples
    duration = len(samples) / from_rate
    target_length = int(duration * to_rate)
    old_indices = np.linspace(0, len(samples) - 1, len(samples))
    new_indices = np.linspace(0, len(samples) - 1, target_length)
    return np.interp(new_indices, old_indices, samples).astype(np.float32)


def extract_vad_trimmed_utterance(audio_8k_float32):
    """
    Feeds the clip through a real CallTurnDetector exactly the way
    two_way_test.py's main loop does — as a sequence of base64-encoded
    PCM16 chunks arriving over time — to extract only what the real
    production pipeline would actually capture and send to Whisper.

    Falls back to manually flushing whatever the detector was still
    holding in its internal buffer if the clip runs out before it ever
    saw enough natural trailing silence to close out a turn on its
    own — confirmed as a real, common case: a short, isolated test
    recording (e.g. someone says "Hello" then the recording just ends)
    often doesn't include a full 1.2s of trailing silence padding after
    the word, so the detector is still correctly mid-utterance right up
    until there's simply no more audio left to feed it. That's not the
    same as "no speech was ever detected at all" (returns None only in
    that genuine case) — it just means the clip ended slightly too soon
    for the detector's own natural completion path.

    Prints exactly when triggering happened, relative to the clip's own
    start — added specifically because an earlier run returned a
    "trimmed" utterance with the SAME duration as the full clip,
    implying the VAD triggered almost immediately rather than genuinely
    skipping leading silence. This makes that visible directly instead
    of only being inferable from the returned duration.
    """
    detector = CallTurnDetector(sample_rate=VAD_SAMPLE_RATE, settings=VAD_SETTINGS)
    pcm_bytes = float32_to_pcm16_bytes(audio_8k_float32)

    # Chunk size matching Exotel's real ~200ms media-event cadence —
    # doesn't need to be exact, add_media_payload buffers internally
    # regardless of how chunks are split.
    chunk_bytes = int(VAD_SAMPLE_RATE * 0.2) * 2  # 200ms, 16-bit samples

    triggered_at_chunk = None
    turn_audio = None
    for chunk_index, i in enumerate(range(0, len(pcm_bytes), chunk_bytes)):
        chunk = pcm_bytes[i:i + chunk_bytes]
        payload = base64.b64encode(chunk).decode()
        was_triggered_before = detector.triggered
        result = detector.add_media_payload(payload)
        if detector.triggered and not was_triggered_before and triggered_at_chunk is None:
            triggered_at_chunk = chunk_index
        if result is not None:
            turn_audio = result
            break  # First completed turn is what we want — the "Hello".

    if triggered_at_chunk is not None:
        triggered_at_seconds = triggered_at_chunk * 0.2
        print(f"  VAD first triggered at ~{triggered_at_seconds:.2f}s into the clip")
        print(f"  (this is how much leading silence, if any, actually got skipped)")
    else:
        print("  VAD never triggered at all during this clip.")

    if turn_audio is not None:
        return pcm16_bytes_to_float32(turn_audio)

    if detector.triggered and detector.speech_frames:
        print("  (clip ended before the VAD saw enough natural trailing")
        print("   silence to close out the turn on its own — manually")
        print("   flushing what it had already accumulated instead.)")
        return pcm16_bytes_to_float32(b"".join(detector.speech_frames))

    return None


async def timed_transcribe(stt, segment, duration_s):
    start = time.perf_counter()
    result = await stt.transcribe_array(segment, duration_s)
    elapsed = time.perf_counter() - start
    return elapsed, result["text"].strip()


async def main():
    print(f"Loading {CLIP_PATH} ...")
    full_clip_16k = load_and_resample(CLIP_PATH, target_sr=WHISPER_SAMPLE_RATE)
    full_duration = len(full_clip_16k) / WHISPER_SAMPLE_RATE
    print(f"Loaded {full_duration:.2f}s of audio (full clip, silence included).\n")

    print("Running the clip through a real CallTurnDetector (at 8kHz, matching")
    print("Exotel's native rate) to extract what production would actually")
    print("capture and send to Whisper...")
    full_clip_8k = resample_float32(full_clip_16k, WHISPER_SAMPLE_RATE, VAD_SAMPLE_RATE)
    trimmed_8k = extract_vad_trimmed_utterance(full_clip_8k)

    if trimmed_8k is None:
        print("ERROR: the VAD never completed a turn on this clip — either the")
        print("speech was too short/quiet to trigger it, or it never saw enough")
        print("trailing silence within the clip to close out the utterance.")
        print("Try a clip with a clearer pause after the word, or check")
        print("VAD_SETTINGS above match what's actually tripping this up.")
        return

    trimmed_16k = resample_float32(trimmed_8k, VAD_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
    trimmed_duration = len(trimmed_16k) / WHISPER_SAMPLE_RATE
    print(f"VAD-trimmed utterance: {trimmed_duration:.2f}s (vs {full_duration:.2f}s full clip)")

    # Saved directly to disk so the EXACT audio actually driving the
    # "trimmed" results can be listened to and verified — added
    # specifically because the VAD can trigger on background noise
    # rather than the real word (see the trigger-timing print above),
    # and the transcript alone doesn't prove which audio actually
    # produced it without hearing it directly.
    import wave
    trimmed_int16 = np.clip(trimmed_16k * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open("trimmed_extract_debug.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(WHISPER_SAMPLE_RATE)
        f.writeframes(trimmed_int16.tobytes())
    print("Saved the exact extracted audio to trimmed_extract_debug.wav —")
    print("listen to this directly to confirm it's genuinely the real \"Hello\"")
    print("and not some other noise that happens to also transcribe that way.")

    # Direct sample-level comparison — added specifically to explain an
    # earlier, genuinely puzzling result: the trimmed variant was ~2x
    # faster than the full clip despite matching durations exactly. If
    # the underlying samples are near-identical, that gap has to come
    # from the 16kHz->8kHz->16kHz resampling round-trip itself changing
    # the audio's technical properties, not from genuinely different
    # content — worth knowing directly rather than assuming either way.
    if len(trimmed_16k) == len(full_clip_16k):
        max_diff = float(np.max(np.abs(trimmed_16k - full_clip_16k)))
        mean_diff = float(np.mean(np.abs(trimmed_16k - full_clip_16k)))
        print(f"Sample-level comparison (same length, so directly comparable):")
        print(f"  max sample difference:  {max_diff:.4f}")
        print(f"  mean sample difference: {mean_diff:.6f}")
        if max_diff < 0.01:
            print("  -> Essentially identical audio. Any timing difference between")
            print("     variants is coming from the resampling round-trip's technical")
            print("     effect on the file, not from genuinely different content.")
        else:
            print("  -> Genuinely different audio content, despite matching duration.")
    else:
        print(f"(Different lengths — {len(full_clip_16k)} vs {len(trimmed_16k)} samples —")
        print(f" so a direct sample-by-sample comparison isn't meaningful here.)")
    print()

    print("Connecting to DeepInfra...")
    stt = DeepInfraSTT(sample_rate=WHISPER_SAMPLE_RATE)
    await stt.verify()
    print("DeepInfra ready.\n")

    full_times, full_texts = [], []
    trimmed_times, trimmed_texts = [], []

    print(f"{'variant':<10} {'rep':<4} {'time':<8} text")
    print("-" * 60)

    for rep in range(1, REPEATS + 1):
        elapsed, text = await timed_transcribe(stt, full_clip_16k, full_duration)
        full_times.append(elapsed)
        full_texts.append(text)
        print(f"{'full':<10} {rep:<4} {elapsed:<8.2f} {text!r}")

    for rep in range(1, REPEATS + 1):
        elapsed, text = await timed_transcribe(stt, trimmed_16k, trimmed_duration)
        trimmed_times.append(elapsed)
        trimmed_texts.append(text)
        print(f"{'trimmed':<10} {rep:<4} {elapsed:<8.2f} {text!r}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Full clip ({full_duration:.2f}s, silence included):")
    print(f"  mean:   {statistics.mean(full_times):.2f}s")
    print(f"  median: {statistics.median(full_times):.2f}s")
    print(f"  texts:  {full_texts}")
    print(f"\nVAD-trimmed ({trimmed_duration:.2f}s, matches real production capture):")
    print(f"  mean:   {statistics.mean(trimmed_times):.2f}s")
    print(f"  median: {statistics.median(trimmed_times):.2f}s")
    print(f"  texts:  {trimmed_texts}")

    print("\nHow to read this:")
    print("- If BOTH variants are slow/inconsistent, the problem is genuinely")
    print("  acoustic — Whisper struggling with a short, isolated word — and no")
    print("  amount of VAD tuning or connection warm-up will fix it.")
    print("- If the trimmed variant is WORSE than the full clip, our own VAD is")
    print("  cutting away real acoustic context Whisper needs, and increasing")
    print("  vad_preroll_ms would be worth trying.")
    print("- If the trimmed variant is BETTER, the full clip's extra silence")
    print("  itself may be confusing Whisper, and our VAD trimming is already")
    print("  helping more than it's hurting here.")


if __name__ == "__main__":
    asyncio.run(main())