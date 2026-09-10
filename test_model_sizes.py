"""
Compares different Whisper model SIZES (base, small, medium), all
using faster-whisper - which is what the pipeline already runs, not
something being newly introduced. An earlier version of this test
mistakenly compared faster-whisper against openai-whisper, assuming
the pipeline was on openai-whisper without actually checking
app/stt/whisper.py directly - it's already on faster-whisper
(WhisperModel, cpu, int8), so that comparison was pointless from the
start. The only real remaining lever for accuracy is model SIZE, not
engine - this tests that directly instead.

No new installs needed - faster_whisper is already a project
dependency (app/stt/whisper.py already imports it).

Usage - point it at the debug_audio_clips folder from dual_tier_stt.py
(or any folder of .wav files, or individual file paths):
    python test_model_sizes.py debug_audio_clips/
    python test_model_sizes.py clip1.wav clip2.wav
"""
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

# medium is a meaningful step up to test - large would likely be too
# slow on CPU to ever be viable for a live call regardless of accuracy,
# so it's left out here rather than testing something that couldn't
# actually be deployed even if it won on accuracy.
MODEL_SIZES = ["base", "small", "medium"]


def get_audio_files(args):
    files = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            files.extend(sorted(p.glob("*.wav")))
        elif p.is_file():
            files.append(p)
    return files


def transcribe(model, audio_path):
    start = time.perf_counter()
    segments, _info = model.transcribe(str(audio_path), beam_size=1)
    text = " ".join(seg.text.strip() for seg in segments)
    elapsed = time.perf_counter() - start
    return text, elapsed


def main():
    audio_files = get_audio_files(sys.argv[1:])
    if not audio_files:
        print("Usage: python test_model_sizes.py <audio_file_or_dir> [...]")
        print("(Run dual_tier_stt.py with SAVE_DEBUG_AUDIO=True during a")
        print("real test call first, to collect genuine telephony audio.)")
        return

    print(f"Found {len(audio_files)} audio file(s) to test.\n")
    print(f"Loading {len(MODEL_SIZES)} model sizes ({', '.join(MODEL_SIZES)})...\n")

    models = {}
    for size in MODEL_SIZES:
        print(f"  Loading {size}...")
        models[size] = WhisperModel(size, device="cpu", compute_type="int8")
    print("All models loaded.\n")

    for audio_path in audio_files:
        print("=" * 70)
        print(f"FILE: {audio_path.name}")
        print("=" * 70)
        for size in MODEL_SIZES:
            text, elapsed = transcribe(models[size], audio_path)
            print(f"[{size:8s}]  {elapsed:5.2f}s  ->  {text!r}")
        print()

    print("=" * 70)
    print("HOW TO READ THIS:")
    print("- Compare the TEXT across sizes for each clip - does a bigger")
    print("  model actually get a mumbled/hard word right where a")
    print("  smaller one didn't?")
    print("- Compare the TIME cost of that improvement - is it still")
    print("  fast enough for a live call? The dual-tier system already")
    print("  only sends SHORT turns to the slower, more accurate tier,")
    print("  so a size bump there costs less overall than it looks like")
    print("  in isolation.")
    print("=" * 70)


if __name__ == "__main__":
    main()