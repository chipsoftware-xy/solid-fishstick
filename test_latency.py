"""
Direct test of two related, real hypotheses about DeepInfra STT latency
in an actual phone call, following up on compare_stt_latency.py's
finding that whichever request "goes first" in a pair pays a real,
consistent ~1.4s penalty over one that immediately follows it:

  1. Does that penalty grow with how long the gap since the last
     request was? In a real call, Spark can speak for 10-20+ seconds
     straight with zero STT calls happening at all during that time —
     if the connection/session genuinely cools down the longer it sits
     idle, nearly every real customer utterance in a live call could be
     paying something like this penalty, not just occasionally.

  2. Does sending a tiny, throwaway "warm-up" request (a fraction of a
     second of near-silent audio) shortly before the real one — timed
     to land while Spark would still be speaking in a real call — bring
     the real request's latency back down? If so, that's a genuinely
     usable, concrete fix: fire a small warm-up ping partway through
     Spark's own reply, so the connection is already warm by the time
     the customer's real speech needs transcribing.

WHY THIS MATTERS: if warm-up genuinely helps, this is a real, low-cost
production fix — one tiny extra request per turn, timed during Spark's
own speech (when the line is otherwise idle anyway), potentially saving
a real ~1-1.4s on every single customer turn in every call.

SETUP: same as compare_stt_latency.py — needs the same sample clip at
audio/input/comparison_sample.wav and settings.yml's DeepInfra key
filled in. Reuses that script's own load_and_resample/split_into_segments
helpers directly rather than reimplementing them.

USAGE:
    python test_stt_gap_effect.py

OUTPUT:
    A table of gap duration vs mean/median latency, with and without a
    warm-up ping, plus a direct warm-up-vs-no-warm-up delta per gap
    duration — the actual number that answers "does warming up during
    Spark's own speech help at all, and by how much."
"""
import asyncio
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from test import load_and_resample, split_into_segments, SAMPLE_CLIP_PATH, WHISPER_SAMPLE_RATE
from app.stt.deepinfra_stt import STT as DeepInfraSTT

# Gap durations to test, in seconds — chosen to span the realistic
# range of how long Spark might speak before the customer's next turn:
# a short deterministic reply on the low end, a long multi-sentence
# reply on the high end (matching the ~12-20s+ pitch durations measured
# on real calls tonight).
GAP_DURATIONS_SECONDS = [0.5, 2, 5, 10, 15, 20]

# How many times to repeat each gap duration (using a different segment
# each time) — averages out the substantial per-call variance already
# confirmed in compare_stt_latency.py, without making this test take
# excessively long. Real DeepInfra calls cost money per request, and
# each repeat here makes 2 (no-warm-up + warm-up), so total calls =
# len(GAP_DURATIONS_SECONDS) * REPEATS_PER_GAP * 2.
REPEATS_PER_GAP = 3

# How long before the real request to fire the warm-up ping — small and
# fixed, meant to represent "a moment before the customer is expected
# to start talking", not a large chunk of the gap itself.
WARMUP_LEAD_SECONDS = 0.3

# A tiny, near-silent clip for the warm-up ping — genuinely minimal
# audio (not meant to be transcribed meaningfully), just enough to be a
# real, complete request that might keep a connection/session warm.
WARMUP_CLIP_SECONDS = 0.3


async def timed_transcribe(stt, segment, duration_s):
    start = time.perf_counter()
    result = await stt.transcribe_array(segment, duration_s)
    elapsed = time.perf_counter() - start
    return elapsed, result["text"].strip()


async def main():
    print(f"Loading {SAMPLE_CLIP_PATH} ...")
    audio = load_and_resample(SAMPLE_CLIP_PATH)
    print(f"Loaded {len(audio) / WHISPER_SAMPLE_RATE:.1f}s of audio.\n")

    # A handful of different segments, cycled through across repeats —
    # deliberately not reusing the exact same audio for every single
    # call, in case DeepInfra does anything (caching, dedup) with
    # identical repeated input that would muddy a pure timing test.
    segments = split_into_segments(audio, WHISPER_SAMPLE_RATE, segment_seconds=3.0, max_segments=10)
    if not segments:
        print("ERROR: no segments extracted from the sample clip.")
        return

    warmup_clip = np.zeros(int(WARMUP_CLIP_SECONDS * WHISPER_SAMPLE_RATE), dtype=np.float32)

    print("Connecting to DeepInfra...")
    stt = DeepInfraSTT(sample_rate=WHISPER_SAMPLE_RATE)
    await stt.verify()
    print("DeepInfra ready.\n")

    print(f"{'gap_s':<7} {'rep':<4} {'no_warmup':<11} {'with_warmup':<12} text")
    print("-" * 80)

    # gap_duration -> list of (no_warmup_time, with_warmup_time) pairs
    results = {gap: [] for gap in GAP_DURATIONS_SECONDS}

    segment_cycle_index = 0

    for gap in GAP_DURATIONS_SECONDS:
        for rep in range(1, REPEATS_PER_GAP + 1):
            segment = segments[segment_cycle_index % len(segments)]
            segment_cycle_index += 1
            seg_duration = len(segment) / WHISPER_SAMPLE_RATE

            # --- No warm-up condition ---
            # Simulates Spark speaking for `gap` seconds with the line
            # otherwise completely idle, then the customer's turn needs
            # transcribing immediately.
            await asyncio.sleep(gap)
            no_warmup_elapsed, no_warmup_text = await timed_transcribe(stt, segment, seg_duration)

            # --- Warm-up condition ---
            # Same total gap, but a tiny ping fires WARMUP_LEAD_SECONDS
            # before the real request — simulating firing a small
            # warm-up request partway through Spark's own reply, timed
            # to land shortly before the customer is expected to speak.
            await asyncio.sleep(max(0.0, gap - WARMUP_LEAD_SECONDS))
            await timed_transcribe(stt, warmup_clip, WARMUP_CLIP_SECONDS)
            with_warmup_elapsed, with_warmup_text = await timed_transcribe(stt, segment, seg_duration)

            results[gap].append((no_warmup_elapsed, with_warmup_elapsed))

            print(f"{gap:<7} {rep:<4} {no_warmup_elapsed:<11.2f} {with_warmup_elapsed:<12.2f} {no_warmup_text!r}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'gap_s':<7} {'no_warmup_mean':<16} {'with_warmup_mean':<18} {'delta':<8}")
    print("-" * 55)

    all_no_warmup, all_with_warmup = [], []
    for gap in GAP_DURATIONS_SECONDS:
        no_warmup_times = [pair[0] for pair in results[gap]]
        with_warmup_times = [pair[1] for pair in results[gap]]
        all_no_warmup.extend(no_warmup_times)
        all_with_warmup.extend(with_warmup_times)
        no_mean = statistics.mean(no_warmup_times)
        warm_mean = statistics.mean(with_warmup_times)
        delta = no_mean - warm_mean
        print(f"{gap:<7} {no_mean:<16.2f} {warm_mean:<18.2f} {delta:+.2f}")

    print(f"\nOverall no-warmup mean:   {statistics.mean(all_no_warmup):.2f}s")
    print(f"Overall with-warmup mean: {statistics.mean(all_with_warmup):.2f}s")
    print(f"Overall warm-up benefit:  {statistics.mean(all_no_warmup) - statistics.mean(all_with_warmup):+.2f}s")

    print("\nHow to read this:")
    print("- If no_warmup_mean clearly RISES as gap_s increases, that directly")
    print("  confirms the 'connection/session cools down over time' hypothesis —")
    print("  meaning a long Spark reply really is making the next STT call slower.")
    print("- If with_warmup_mean stays LOW and roughly flat across all gap")
    print("  durations while no_warmup_mean rises, that confirms a small warm-up")
    print("  ping fired during Spark's own speech is a real, usable fix.")
    print("- If both columns look similar at every gap duration, the earlier")
    print("  'goes first' penalty likely wasn't about idle time at all, and this")
    print("  specific fix wouldn't help — worth knowing before building it into")
    print("  production for no real benefit.")


if __name__ == "__main__":
    asyncio.run(main())