import time
import wave
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

# Audio processing settings
TARGET_PEAK = 24000
PADDING_SECONDS = 0.30


def record_until_enter(output_path: str):
    """
    Record from the default microphone until Enter is pressed.

    The recording is:
    - captured at 16 kHz mono
    - gain-normalized to improve Whisper recognition
    - padded with silence at the beginning and end

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

    if not frames:
        print("[MIC] No audio captured.")
        return 0.0

    # Combine recorded chunks
    audio = np.concatenate(frames, axis=0).flatten()

    # ---------------------------------------------------------
    # AUDIO DIAGNOSTICS
    # ---------------------------------------------------------

    peak_before = int(np.max(np.abs(audio)))

    rms_before = float(
        np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    )

    print(f"[MIC] Samples: {len(audio)}")
    print(f"[MIC] Peak before gain: {peak_before}")
    print(f"[MIC] RMS before gain: {rms_before:.0f}")
    # recording.py
    def trim_silence(audio: np.ndarray, sample_rate: int,
                     frame_ms: int = 30, rms_threshold: float = 300):
         """
    Returns (trimmed_audio, had_speech).
    Finds the first/last frame whose RMS exceeds rms_threshold and
    trims to that span (+/- a little padding). If no frame exceeds
    the threshold, the recording is pure silence/noise — had_speech
    is False and trimmed_audio is empty.

    rms_threshold is a starting point — print rms_before on a few
    real clips (quiet room vs. you speaking) and tune it to sit
    clearly between your noise floor and your voice.
    """
         frame_len = int(sample_rate * frame_ms / 1000)
         n_frames = len(audio) // frame_len

         speech_frame_idxs = []
         for i in range(n_frames):
             frame = audio[i * frame_len:(i + 1) * frame_len]
             rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
             if rms > rms_threshold:
                 speech_frame_idxs.append(i)
         if not speech_frame_idxs:
             return np.array([], dtype=np.int16), False
         pad_frames = 3  # ~90ms of context around detected speech
         start = max(0, speech_frame_idxs[0] - pad_frames) * frame_len
         end = min(n_frames, speech_frame_idxs[-1] + 1 + pad_frames) * frame_len

         return audio[start:end], True

    audio=np.concatenate(frames, axis=0).flatten()
    audio, had_speech=trim_silence(audio, SAMPLE_RATE)
    if not had_speech:
        print("[MIC] No speech detected - skipping.")
        return duration, False
    peak_before=int(np.max(np.abs(audio)))

    # ---------------------------------------------------------
    # GAIN NORMALIZATION
    # ---------------------------------------------------------

    if peak_before > 0:

        # Calculate gain required to reach target peak
        gain = TARGET_PEAK / peak_before

        # Don't reduce volume if the recording is already loud.
        gain = max(1.0, gain)

        audio_float = audio.astype(np.float32) * gain

        # Prevent int16 clipping
        audio_float = np.clip(
            audio_float,
            -32768,
            32767
        )

        audio = audio_float.astype(np.int16)

    peak_after = int(np.max(np.abs(audio)))

    rms_after = float(
        np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    )

    print(f"[MIC] Gain applied: {gain:.2f}x")
    print(f"[MIC] Peak after gain: {peak_after}")
    print(f"[MIC] RMS after gain: {rms_after:.0f}")

    # ---------------------------------------------------------
    # SILENCE PADDING
    # ---------------------------------------------------------

    padding_samples = int(
        SAMPLE_RATE * PADDING_SECONDS
    )

    padding = np.zeros(
        padding_samples,
        dtype=np.int16
    )

    audio = np.concatenate([
        padding,
        audio,
        padding
    ])

    # ---------------------------------------------------------
    # SAVE WAV
    # ---------------------------------------------------------

    with wave.open(output_path, "wb") as wav:

        wav.setnchannels(CHANNELS)

        wav.setsampwidth(2)  # int16

        wav.setframerate(SAMPLE_RATE)

        wav.writeframes(audio.tobytes())

    print(f"[MIC] Recorded: {duration:.2f}s")
    print(f"[MIC] Final audio: {len(audio) / SAMPLE_RATE:.2f}s")
    print(f"[MIC] Saved: {output_path}")

    return duration, True