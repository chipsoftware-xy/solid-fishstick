"""
Direct, empirical A/B/C comparison: local faster-whisper (CPU, int8) vs
DeepInfra-hosted whisper-large-v3-turbo with service_tier="priority" vs
the same DeepInfra model on its default tier — built specifically to
settle whether STT latency is genuinely a model-speed problem, a
network/queuing-overhead problem, or something the "priority" tier
parameter actually fixes, rather than continuing to guess from
production call logs alone.

WHY THIS MATTERS: an earlier run of this same script already showed
DeepInfra (2.78s mean) genuinely beating local inference (6.61s mean)
on this hardware — overturning an initial hypothesis that local would
be faster. But that DeepInfra number already had service_tier="priority"
applied, with no baseline to compare it against — this version adds
that baseline directly, sending the SAME identical audio segment
through DeepInfra twice (once per tier), so the only variable that
differs between those two calls is the tier itself, not the audio, not
network conditions at a different moment, not anything else.

SETUP:
    pip install faster-whisper soundfile numpy --break-system-packages

    Place your ~4-minute sample clip at: audio/input/comparison_sample.wav
    (any format soundfile can read — wav, flac, ogg; mono or stereo,
    any sample rate — this script resamples to 16kHz to match both
    engines' expected input)

    Make sure settings.yml's stt.deepinfra.api_key is filled in (same
    key two_way_test.py already uses) — this script reuses
    app/stt/deepinfra_stt.py's own STT class directly, not a
    reimplementation, so it's testing the exact same code path the
    real pipeline uses.

USAGE:
    python compare_stt_latency.py

    Optionally edit LOCAL_MODEL_SIZE below to compare different local
    model choices across separate runs — distil-large-v3 (closest to
    DeepInfra's turbo quality-wise), medium.en (middle ground), or
    distil-medium.en (fastest, most quality loss). No need to redownload
    between runs — faster-whisper caches each model size separately.

    Note: this version makes 2 DeepInfra calls per segment instead of 1
    (once per tier), so a full run takes longer and costs roughly double
    the DeepInfra usage of the earlier, single-tier version — reduce
    MAX_SEGMENTS below if that's a concern.

OUTPUT:
    A segment-by-segment table (duration, local time, DeepInfra-priority
    time, DeepInfra-default time, plus text from local and DeepInfra-
    priority) plus summary statistics for all three, including a direct
    priority-vs-default delta — so you can see exactly how much (if
    anything) the tier parameter is actually buying, separately from
    judging accuracy across engines.
"""
import asyncio
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# ---- Configuration ----
SAMPLE_CLIP_PATH = "audio/input/comparison_sample.wav"
SEGMENT_LENGTH_SECONDS = 3.0  # Roughly matches typical short call utterances
                              # ("yes", "okay", a name) — the case where
                              # the ~2s overhead is proportionally worst.
MAX_SEGMENTS = 20  # Caps a 4-minute clip at a reasonable number of test
                   # calls rather than every possible 3s slice — DeepInfra
                   # calls cost money per request.

# Try distil-large-v3 first (closest match to DeepInfra's turbo, per the
# suggested tradeoff) — swap to "medium.en" or "distil-medium.en" for a
# separate comparison run if this is still too slow on your CPU.
LOCAL_MODEL_SIZE = "distil-large-v3"
LOCAL_COMPUTE_TYPE = "int8"

WHISPER_SAMPLE_RATE = 16000


def load_and_resample(path, target_sr=WHISPER_SAMPLE_RATE):
    """Loads the sample clip and resamples to 16kHz mono float32 —
    matching what both engines actually expect, so neither one is
    disadvantaged by a format mismatch that has nothing to do with
    their real transcription speed."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Downmix to mono if stereo.
    if sr != target_sr:
        # Simple linear resampling — adequate for this comparison,
        # avoiding a heavier dependency for one conversion. Matches the
        # same approach already used elsewhere in this codebase
        # (audio_utils.py's resample_pcm16).
        duration = len(audio) / sr
        target_length = int(duration * target_sr)
        old_indices = np.linspace(0, len(audio) - 1, len(audio))
        new_indices = np.linspace(0, len(audio) - 1, target_length)
        audio = np.interp(new_indices, old_indices, audio).astype(np.float32)
    return audio


def split_into_segments(audio, sample_rate, segment_seconds, max_segments):
    """Splits into fixed-length chunks — deliberately simple (not
    silence-aware VAD splitting) since the goal here is comparing raw
    transcription speed on short clips, not perfecting utterance
    boundaries. A segment landing mid-word occasionally is fine for a
    speed comparison; it would matter more for an accuracy-only test."""
    segment_samples = int(segment_seconds * sample_rate)
    segments = []
    for start in range(0, len(audio), segment_samples):
        segment = audio[start:start + segment_samples]
        if len(segment) < sample_rate * 0.5:
            continue  # Skip a trailing sliver under 0.5s — not representative.
        segments.append(segment)
        if len(segments) >= max_segments:
            break
    return segments


def transcribe_local(model, audio_segment):
    start = time.perf_counter()
    segments, _info = model.transcribe(audio_segment, language="en", beam_size=5)
    # Whisper segments already carry their own leading space, so joining
    # with "" (not " ") avoids doubled spaces between segments —
    # confirmed directly via a test: " ".join(...) produced
    # "Hello  world" (double space) on real segment text shapes.
    text = "".join(seg.text for seg in segments).strip()
    elapsed = time.perf_counter() - start
    return elapsed, text


async def transcribe_deepinfra(stt, audio_segment, duration_s, service_tier="priority"):
    start = time.perf_counter()
    result = await stt.transcribe_array(audio_segment, duration_s, service_tier=service_tier)
    elapsed = time.perf_counter() - start
    return elapsed, result["text"].strip()


async def main():
    clip_path = Path(SAMPLE_CLIP_PATH)
    if not clip_path.exists():
        print(f"ERROR: {SAMPLE_CLIP_PATH} not found.")
        print(f"Place your ~4-minute sample clip there (any format soundfile can read) and re-run.")
        return

    print(f"Loading {SAMPLE_CLIP_PATH} ...")
    audio = load_and_resample(str(clip_path))
    total_duration = len(audio) / WHISPER_SAMPLE_RATE
    print(f"Loaded {total_duration:.1f}s of audio at {WHISPER_SAMPLE_RATE}Hz.")

    segments = split_into_segments(audio, WHISPER_SAMPLE_RATE, SEGMENT_LENGTH_SECONDS, MAX_SEGMENTS)
    print(f"Split into {len(segments)} segments of ~{SEGMENT_LENGTH_SECONDS}s each.\n")

    print(f"Loading local model ({LOCAL_MODEL_SIZE}, compute_type={LOCAL_COMPUTE_TYPE})...")
    print("(First run downloads the model — subsequent runs use the cached copy.)")
    from faster_whisper import WhisperModel
    local_model = WhisperModel(LOCAL_MODEL_SIZE, device="cpu", compute_type=LOCAL_COMPUTE_TYPE)
    print("Local model loaded.\n")

    print("Connecting to DeepInfra...")
    sys.path.insert(0, ".")
    from app.stt.deepinfra_stt import STT as DeepInfraSTT
    deepinfra_stt = DeepInfraSTT(sample_rate=WHISPER_SAMPLE_RATE)
    await deepinfra_stt.verify()
    print("DeepInfra ready.\n")

    local_times, deepinfra_priority_times, deepinfra_default_times = [], [], []
    went_first_times, went_second_times = [], []

    print(f"{'#':<3} {'dur':<6} {'local_t':<9} {'di_prio':<9} {'di_def':<9} {'order':<12} local_text / di_priority_text")
    print("-" * 110)

    for i, segment in enumerate(segments, 1):
        seg_duration = len(segment) / WHISPER_SAMPLE_RATE

        local_elapsed, local_text = transcribe_local(local_model, segment)
        local_times.append(local_elapsed)

        # Same identical segment, sent twice — once per tier — so the
        # only variable that differs between these two calls is the
        # tier itself, not the audio, not network conditions at a
        # different moment, not anything else.
        #
        # Order alternated by segment index — a real, necessary fix,
        # not paranoia: priority was previously ALWAYS called first and
        # default ALWAYS second, every single segment. If there's any
        # generic effect where the second call in a quick pair benefits
        # from something the first doesn't (connection reuse, session
        # affinity, DeepInfra-side request scheduling), that would make
        # default look faster every time regardless of whether the tier
        # itself matters at all — deepinfra_stt.verify() warming the
        # connection before the loop even starts only rules out "very
        # first call in the whole script is slow", not this specific,
        # repeating per-segment positional effect. Alternating spreads
        # any such effect evenly across both tiers instead of
        # systematically favoring one.
        if i % 2 == 1:
            di_prio_elapsed, di_prio_text = await transcribe_deepinfra(deepinfra_stt, segment, seg_duration, service_tier="priority")
            di_def_elapsed, di_def_text = await transcribe_deepinfra(deepinfra_stt, segment, seg_duration, service_tier=None)
        else:
            di_def_elapsed, di_def_text = await transcribe_deepinfra(deepinfra_stt, segment, seg_duration, service_tier=None)
            di_prio_elapsed, di_prio_text = await transcribe_deepinfra(deepinfra_stt, segment, seg_duration, service_tier="priority")

        deepinfra_priority_times.append(di_prio_elapsed)
        deepinfra_default_times.append(di_def_elapsed)

        # Tracked independent of which tier it was — this is what
        # actually reveals whether "goes first vs goes second" matters
        # at all, regardless of tier.
        if i % 2 == 1:
            went_first_times.append(di_prio_elapsed)
            went_second_times.append(di_def_elapsed)
        else:
            went_first_times.append(di_def_elapsed)
            went_second_times.append(di_prio_elapsed)

        _order_note = "(prio->def)" if i % 2 == 1 else "(def->prio)"
        print(f"{i:<3} {seg_duration:<6.1f} {local_elapsed:<9.2f} {di_prio_elapsed:<9.2f} {di_def_elapsed:<9.2f} {_order_note:<12} {local_text!r} / {di_prio_text!r}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Segments tested: {len(segments)}")
    print(f"\nLocal ({LOCAL_MODEL_SIZE}, {LOCAL_COMPUTE_TYPE}):")
    print(f"  mean:   {statistics.mean(local_times):.2f}s")
    print(f"  median: {statistics.median(local_times):.2f}s")
    print(f"  min:    {min(local_times):.2f}s")
    print(f"  max:    {max(local_times):.2f}s")
    print(f"\nDeepInfra, service_tier=priority:")
    print(f"  mean:   {statistics.mean(deepinfra_priority_times):.2f}s")
    print(f"  median: {statistics.median(deepinfra_priority_times):.2f}s")
    print(f"  min:    {min(deepinfra_priority_times):.2f}s")
    print(f"  max:    {max(deepinfra_priority_times):.2f}s")
    print(f"\nDeepInfra, default tier (no service_tier set):")
    print(f"  mean:   {statistics.mean(deepinfra_default_times):.2f}s")
    print(f"  median: {statistics.median(deepinfra_default_times):.2f}s")
    print(f"  min:    {min(deepinfra_default_times):.2f}s")
    print(f"  max:    {max(deepinfra_default_times):.2f}s")
    _prio_mean = statistics.mean(deepinfra_priority_times)
    _def_mean = statistics.mean(deepinfra_default_times)
    _delta = _def_mean - _prio_mean
    print(f"\npriority vs default delta: {_delta:+.2f}s mean ({'priority faster' if _delta > 0 else 'default faster or no difference'})")

    print(f"\nOrder-effect check (regardless of which tier went first/second):")
    print(f"  went first  — mean: {statistics.mean(went_first_times):.2f}s  median: {statistics.median(went_first_times):.2f}s")
    print(f"  went second — mean: {statistics.mean(went_second_times):.2f}s  median: {statistics.median(went_second_times):.2f}s")
    _order_delta = statistics.mean(went_first_times) - statistics.mean(went_second_times)
    print(f"  first-vs-second delta: {_order_delta:+.2f}s")
    print("  If this delta is large, position in the pair (not the tier) is doing")
    print("  most of the work in the priority-vs-default numbers above — the tier")
    print("  comparison would need to be re-run with even more segments, or with")
    print("  each tier tested in a fully separate run, to trust it.")

    print("\nCompare the local_text / di_priority_text columns above for accuracy —")
    print("a faster engine that also transcribes worse isn't actually a win.")


if __name__ == "__main__":
    asyncio.run(main())