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
# each repeat here now makes 3 real-transcribe calls (no-warmup +
# early-warmup + late-warmup) plus 2 warm-up pings, so total real
# transcribe calls = len(GAP_DURATIONS_SECONDS) * REPEATS_PER_GAP * 3.
REPEATS_PER_GAP = 3

# --- Two competing warm-up timing hypotheses being distinguished here ---
#
# "late" was the only thing the original script tested: fire the ping
# a fixed, small lead time before the real request — i.e. shortly
# before the customer is expected to speak, regardless of how long the
# gap is. This is the evidence-backed condition from earlier runs.
#
# "early" is the untested hypothesis: fire a single ping right as the
# gap starts (i.e. right as Spark begins a long reply) and then do
# nothing until the real request lands, however long the gap turns
# out to be. If DeepInfra's connection/session "cools down" over
# time, an early-only ping should stop helping (or help less) as the
# gap grows toward 15-20s. If it stays just as effective as "late" at
# every gap length, that's evidence the warm-up effect isn't time-
# sensitive and firing once at the start of Spark's reply is fine.
#
# Comparing early vs late vs no-warmup at each gap duration directly
# tells you whether "fire once at the start" holds up, without having
# to guess from the no-warmup-only trend.

# Lead time before the real request for the "late" condition — small
# and fixed, meant to represent "a moment before the customer is
# expected to start talking", not a large chunk of the gap itself.
WARMUP_LEAD_SECONDS = 0.3

# A tiny, near-silent clip for the warm-up pings — genuinely minimal
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

    print(f"{'gap_s':<7} {'rep':<4} {'no_warmup':<11} {'early_warmup':<13} {'late_warmup':<12} text")
    print("-" * 90)

    # gap_duration -> list of (no_warmup_time, early_warmup_time, late_warmup_time)
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

            # --- Early warm-up condition ---
            # Ping fires immediately (t=0 of the gap), representing
            # "as Spark starts talking", then nothing happens until
            # the real request lands `gap` seconds later. This is the
            # untested hypothesis: does a single early ping keep the
            # connection warm all the way out to 15-20s, or does the
            # benefit fade with elapsed time?
            await timed_transcribe(stt, warmup_clip, WARMUP_CLIP_SECONDS)
            await asyncio.sleep(gap)
            early_warmup_elapsed, early_warmup_text = await timed_transcribe(stt, segment, seg_duration)

            # --- Late warm-up condition ---
            # Same total gap, but the ping fires WARMUP_LEAD_SECONDS
            # before the real request — simulating a small warm-up
            # request timed to land shortly before the customer is
            # expected to speak. This is the condition the original
            # script already had evidence for.
            await asyncio.sleep(max(0.0, gap - WARMUP_LEAD_SECONDS))
            await timed_transcribe(stt, warmup_clip, WARMUP_CLIP_SECONDS)
            await asyncio.sleep(WARMUP_LEAD_SECONDS)
            late_warmup_elapsed, late_warmup_text = await timed_transcribe(stt, segment, seg_duration)

            results[gap].append((no_warmup_elapsed, early_warmup_elapsed, late_warmup_elapsed))

            print(
                f"{gap:<7} {rep:<4} {no_warmup_elapsed:<11.2f} "
                f"{early_warmup_elapsed:<13.2f} {late_warmup_elapsed:<12.2f} {no_warmup_text!r}"
            )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'gap_s':<7} {'no_warmup_mean':<16} {'early_warmup_mean':<19} {'late_warmup_mean':<18} "
          f"{'early_delta':<12} {'late_delta':<10}")
    print("-" * 90)

    all_no_warmup, all_early_warmup, all_late_warmup = [], [], []
    for gap in GAP_DURATIONS_SECONDS:
        no_warmup_times = [triple[0] for triple in results[gap]]
        early_warmup_times = [triple[1] for triple in results[gap]]
        late_warmup_times = [triple[2] for triple in results[gap]]
        all_no_warmup.extend(no_warmup_times)
        all_early_warmup.extend(early_warmup_times)
        all_late_warmup.extend(late_warmup_times)

        no_mean = statistics.mean(no_warmup_times)
        early_mean = statistics.mean(early_warmup_times)
        late_mean = statistics.mean(late_warmup_times)
        early_delta = no_mean - early_mean
        late_delta = no_mean - late_mean

        print(f"{gap:<7} {no_mean:<16.2f} {early_mean:<19.2f} {late_mean:<18.2f} "
              f"{early_delta:+<12.2f} {late_delta:+<10.2f}")

    print(f"\nOverall no-warmup mean:    {statistics.mean(all_no_warmup):.2f}s")
    print(f"Overall early-warmup mean: {statistics.mean(all_early_warmup):.2f}s")
    print(f"Overall late-warmup mean:  {statistics.mean(all_late_warmup):.2f}s")

    print("\nHow to read this:")
    print("- If no_warmup_mean clearly RISES as gap_s increases, that directly")
    print("  confirms the 'connection/session cools down over time' hypothesis —")
    print("  meaning a long Spark reply really is making the next STT call slower.")
    print("- late_warmup_mean staying LOW and roughly flat across all gap durations")
    print("  is the evidence-backed result from before: a ping fired shortly before")
    print("  the customer is expected to speak is a real, usable fix.")
    print("- The open question this version answers: does early_warmup_mean ALSO")
    print("  stay low and flat out to 15-20s, tracking late_warmup_mean? If so,")
    print("  'fire once at the start of Spark's reply' is safe to build. If")
    print("  early_warmup_mean instead drifts back up toward no_warmup_mean as gap_s")
    print("  grows (i.e. the early/late gap widens at 15-20s), the warm-up effect")
    print("  decays over time and only firing shortly before the real request works.")
    print("- If neither warmup column beats no_warmup at any gap duration, the")
    print("  earlier 'goes first' penalty likely wasn't about idle time at all.")


if __name__ == "__main__":
    asyncio.run(main())