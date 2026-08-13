import time
import wave
import sounddevice as sd


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


def record_until_enter(output_path: str):
    """
    Record from the default microphone until Enter is pressed.

    Returns:
        recording_duration
    """

    print("\nPress ENTER to start recording...")
    input()

    print("🎤 Listening... Press ENTER again to stop.")

    start = time.perf_counter()

    frames = []

    def callback(indata, frames_count, time_info, status):
        if status:
            print(f"[MIC] {status}")

        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
    ):
        input()

    duration = time.perf_counter() - start

    audio_data = b"".join(frame.tobytes() for frame in frames)

    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)  # int16
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_data)

    print(f"[MIC] Recorded: {duration:.2f}s")
    print(f"[MIC] Saved: {output_path}")

    return duration