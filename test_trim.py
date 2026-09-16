import asyncio
import time
import soundfile as sf

from app.stt.deepinfra_stt import STT

#not used yet
AUDIO_FILE = "audio/input/yes1.wav"
REPEATS = 10


async def main():
    print(f"Loading {AUDIO_FILE}...")

    audio, sr = sf.read(AUDIO_FILE, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        raise ValueError(f"Expected 16000 Hz, got {sr} Hz")

    duration = len(audio) / sr

    print(f"Duration: {duration:.2f}s")
    print(f"Running the SAME clip {REPEATS} times.\n")

    print("Connecting to DeepInfra...")
    stt = STT()

    await stt.initialize()

    print("DeepInfra ready.\n")

    times = []

    for i in range(REPEATS):
        start = time.perf_counter()

        text = await stt.transcribe_array(
            audio,
            duration_s=duration
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(
            f"Run {i+1:2d}: "
            f"{elapsed:.3f}s   "
            f"{text!r}"
        )

    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)

    print(f"Mean:   {sum(times) / len(times):.3f}s")
    print(f"Min:    {min(times):.3f}s")
    print(f"Max:    {max(times):.3f}s")

    sorted_times = sorted(times)
    print(f"Median: {sorted_times[len(times)//2]:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())