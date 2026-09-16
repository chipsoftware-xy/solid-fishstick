import asyncio
import random
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from test import load_and_resample, split_into_segments, SAMPLE_CLIP_PATH, WHISPER_SAMPLE_RATE
from app.stt.deepinfra_stt import STT as DeepInfraSTT

# --- Cost knobs -------------------------------------------------------
# Every value here directly multiplies the number of billed DeepInfra
# calls. Trimmed down from the first version, which tested 6 gaps x 3
# reps x (3 real transcribes + 2 pings) = 90 calls per run. Bump these
# back up once you're confident the design is clean and want tighter
# confidence intervals.

# Gap durations to test, in seconds. Cut from 6 points to 4: 0.5s and
# 2s already looked flat and uninformative in the last run (early and
# late both roughly matched no_warmup), so they're dropped in favor of
# keeping the points that actually separate the hypotheses — the short
# end as a sanity check, and the region (5-20s) where early_warmup
# diverged from late_warmup.
GAP_DURATIONS_SECONDS = [0.5, 5, 10, 20]

# Repeats per (gap, condition). Lower than before on purpose — this is
# a cost/precision tradeoff. Raise it back to 3+ for a confirmation run
# once the cheap run points at a clear pattern.
REPEATS_PER_GAP = 2

# Real transcribe audio length per segment. Shortened from 3.0s; still
# long enough to be a realistic "customer's turn" but cuts the biggest
# per-call audio-duration cost by ~33%.
SEGMENT_SECONDS = 2.0

# Free — no API call, no cost. Idle time inserted between reps so a
# previous rep's warm-up pings can't quietly leave the connection warm
# for the next rep's calls (this was the likely confound flagged after
# the last run). Only fires between reps, not between the 3 conditions
# within a rep — those still need to happen back-to-back on the same
# gap timeline by design.
COOLDOWN_BETWEEN_REPS_SECONDS = 30

# --- Warm-up ping design ----------------------------------------------
# Lead time before the real request for the "late" condition.
WARMUP_LEAD_SECONDS = 0.3

# Tiny, near-silent clip for the warm-up pings.
WARMUP_CLIP_SECONDS = 0.3

CONDITIONS = ["no_warmup", "early_warmup", "late_warmup"]


async def timed_transcribe(stt, segment, duration_s):
    start = time.perf_counter()
    result = await stt.transcribe_array(segment, duration_s)
    elapsed = time.perf_counter() - start
    return elapsed, result["text"].strip()


async def run_condition(stt, condition, gap, segment, seg_duration, warmup_clip):
    """Run a single condition for a single gap and return (elapsed, text)."""
    if condition == "no_warmup":
        await asyncio.sleep(gap)
        return await timed_transcribe(stt, segment, seg_duration)

    if condition == "early_warmup":
        # Ping fires immediately (t=0 of the gap) — "as Spark starts
        # talking" — then nothing until the real request lands `gap`
        # seconds later. Untested hypothesis: does one early ping keep
        # the connection warm all the way to 15-20s?
        await timed_transcribe(stt, warmup_clip, WARMUP_CLIP_SECONDS)
        await asyncio.sleep(gap)
        return await timed_transcribe(stt, segment, seg_duration)

    if condition == "late_warmup":
        # Ping fires WARMUP_LEAD_SECONDS before the real request —
        # the evidence-backed condition from the original script.
        await asyncio.sleep(max(0.0, gap - WARMUP_LEAD_SECONDS))
        await timed_transcribe(stt, warmup_clip, WARMUP_CLIP_SECONDS)
        await asyncio.sleep(WARMUP_LEAD_SECONDS)
        return await timed_transcribe(stt, segment, seg_duration)

    raise ValueError(f"unknown condition: {condition}")


def calls_per_rep(condition):
    return 2 if condition in ("early_warmup", "late_warmup") else 1


async def main():
    print(f"Loading {SAMPLE_CLIP_PATH} ...")
    audio = load_and_resample(SAMPLE_CLIP_PATH)
    print(f"Loaded {len(audio) / WHISPER_SAMPLE_RATE:.1f}s of audio.\n")

    segments = split_into_segments(audio, WHISPER_SAMPLE_RATE, segment_seconds=SEGMENT_SECONDS, max_segments=10)
    if not segments:
        print("ERROR: no segments extracted from the sample clip.")
        return

    warmup_clip = np.zeros(int(WARMUP_CLIP_SECONDS * WHISPER_SAMPLE_RATE), dtype=np.float32)

    total_calls = len(GAP_DURATIONS_SECONDS) * REPEATS_PER_GAP * sum(calls_per_rep(c) for c in CONDITIONS)
    print(f"Planned DeepInfra calls this run: {total_calls} "
          f"({len(GAP_DURATIONS_SECONDS)} gaps x {REPEATS_PER_GAP} reps x 3 conditions)")
    print("Adjust GAP_DURATIONS_SECONDS / REPEATS_PER_GAP at the top of this file to change that.\n")

    print("Connecting to DeepInfra...")
    stt = DeepInfraSTT(sample_rate=WHISPER_SAMPLE_RATE)
    await stt.verify()
    print("DeepInfra ready.\n")

    print(f"{'gap_s':<7} {'rep':<4} {'no_warmup':<11} {'early_warmup':<13} {'late_warmup':<12} text")
    print("-" * 90)

    # gap_duration -> list of (no_warmup_time, early_warmup_time, late_warmup_time)
    results = {gap: [] for gap in GAP_DURATIONS_SECONDS}

    segment_cycle_index = 0
    first_rep = True

    for gap in GAP_DURATIONS_SECONDS:
        for rep in range(1, REPEATS_PER_GAP + 1):
            segment = segments[segment_cycle_index % len(segments)]
            segment_cycle_index += 1
            seg_duration = len(segment) / WHISPER_SAMPLE_RATE

            if not first_rep:
                # No API call here — just idle time so the previous
                # rep's last warm-up ping can't bleed into this rep's
                # results, regardless of which condition runs first.
                await asyncio.sleep(COOLDOWN_BETWEEN_REPS_SECONDS)
            first_rep = False

            # Randomize which of the 3 conditions goes first/second/
            # third this rep, so no single condition systematically
            # inherits warmth left over from the one before it.
            order = CONDITIONS[:]
            random.shuffle(order)

            elapsed_by_condition = {}
            text_by_condition = {}
            for condition in order:
                elapsed, text = await run_condition(stt, condition, gap, segment, seg_duration, warmup_clip)
                elapsed_by_condition[condition] = elapsed
                text_by_condition[condition] = text

            no_warmup_elapsed = elapsed_by_condition["no_warmup"]
            early_warmup_elapsed = elapsed_by_condition["early_warmup"]
            late_warmup_elapsed = elapsed_by_condition["late_warmup"]
            results[gap].append((no_warmup_elapsed, early_warmup_elapsed, late_warmup_elapsed))

            print(
                f"{gap:<7} {rep:<4} {no_warmup_elapsed:<11.2f} "
                f"{early_warmup_elapsed:<13.2f} {late_warmup_elapsed:<12.2f} "
                f"order={order} {text_by_condition['no_warmup']!r}"
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
              f"{early_delta:<+12.2f} {late_delta:<+10.2f}")

    print(f"\nOverall no-warmup mean:    {statistics.mean(all_no_warmup):.2f}s")
    print(f"Overall early-warmup mean: {statistics.mean(all_early_warmup):.2f}s")
    print(f"Overall late-warmup mean:  {statistics.mean(all_late_warmup):.2f}s")

    print("\nHow to read this:")
    print("- If no_warmup_mean clearly RISES as gap_s increases, that directly")
    print("  confirms the 'connection/session cools down over time' hypothesis.")
    print("- late_warmup_mean staying LOW and roughly flat across all gap durations")
    print("  is the evidence-backed result: a ping fired shortly before the")
    print("  customer is expected to speak is a real, usable fix.")
    print("- Open question: does early_warmup_mean ALSO stay low and flat out to")
    print("  15-20s, tracking late_warmup_mean? If it instead drifts back up")
    print("  toward no_warmup_mean as gap_s grows, the warm-up effect decays over")
    print("  time and only firing shortly before the real request works.")
    print("- Each row's 'order=' in the raw log lets you check for residual")
    print("  order effects within a rep — e.g. whether whichever condition ran")
    print("  first in a given rep tends to run slower regardless of which one it was.")


if __name__ == "__main__":
    asyncio.run(main())