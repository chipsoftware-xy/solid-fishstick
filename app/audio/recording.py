import time
import wave
from collections import deque

import sounddevice as sd
import numpy as np
import webrtcvad

from app.config import settings

_AUDIO_SETTINGS = settings["audio"]

SAMPLE_RATE = _AUDIO_SETTINGS["sample_rate"]
CHANNELS = 1
DTYPE = "int16"

# Target peak amplitude after normalization, out of a possible 32767 for
# int16 audio. Left with headroom below the true max to avoid clipping.
TARGET_PEAK = _AUDIO_SETTINGS["target_peak"]

# Cap on how much a quiet recording gets amplified. Without a cap, a
# near-silent clip (peak close to 0) would get amplified enormously,
# mostly boosting noise floor rather than real signal.
MAX_GAIN = _AUDIO_SETTINGS["max_gain"]

# Silence padding added to both ends of the recording, in milliseconds.
# Recording starts/stops the instant Enter is pressed, so there's a real
# chance actual speech gets clipped right at the start or end — Whisper
# is known to mishandle words with no lead-in silence around them.
PAD_MS = _AUDIO_SETTINGS["silence_pad_ms"]


def _normalize_gain(audio_array: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio_array)) if audio_array.size else 0
    if peak == 0:
        return audio_array

    gain = min(TARGET_PEAK / peak, MAX_GAIN)
    amplified = audio_array.astype(np.float32) * gain
    return np.clip(amplified, -32768, 32767).astype(np.int16)


def _pad_silence(audio_array: np.ndarray, sample_rate: int, pad_ms: int = PAD_MS) -> np.ndarray:
    pad_samples = int(sample_rate * pad_ms / 1000)
    silence = np.zeros(pad_samples, dtype=np.int16)
    return np.concatenate([silence, audio_array, silence])


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
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    raw_peak = int(np.max(np.abs(audio_array))) if audio_array.size else 0
    raw_rms = (
        float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)))
        if audio_array.size else 0.0
    )

    audio_array = _normalize_gain(audio_array)
    audio_array = _pad_silence(audio_array, SAMPLE_RATE)
    audio_data = audio_array.tobytes()

    print(f"[MIC] Samples: {len(audio_array)}")
    print(f"[MIC] Peak (raw -> normalized): {raw_peak} -> {int(np.max(np.abs(audio_array)))}")
    print(f"[MIC] RMS (raw -> normalized): {raw_rms:.0f} -> "
          f"{np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)):.0f}")

    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)  # int16
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_data)

    print(f"[MIC] Recorded: {duration:.2f}s")
    print(f"[MIC] Saved: {output_path}")

    return duration


def _finalize_and_write(frames, output_path):
    """Shared with record_until_enter: gain-normalize, pad, and save."""
    audio_data = b"".join(frames)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    raw_peak = int(np.max(np.abs(audio_array))) if audio_array.size else 0
    raw_rms = (
        float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)))
        if audio_array.size else 0.0
    )

    audio_array = _normalize_gain(audio_array)
    audio_array = _pad_silence(audio_array, SAMPLE_RATE)
    audio_data = audio_array.tobytes()

    print(f"[MIC] Samples: {len(audio_array)}")
    print(f"[MIC] Peak (raw -> normalized): {raw_peak} -> {int(np.max(np.abs(audio_array)))}")
    print(f"[MIC] RMS (raw -> normalized): {raw_rms:.0f} -> "
          f"{np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)):.0f}")

    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_data)

    print(f"[MIC] Saved: {output_path}")


def record_on_voice_activity(output_path: str):
    """
    Listens continuously and automatically records one utterance —
    starts when speech is detected, stops after a sustained silence.
    No keypress needed. Uses webrtcvad (Google's WebRTC voice activity
    detector) frame-by-frame on the live mic stream.

    All the vad_* settings below are untested starting points — expect
    to re-tune against your actual mic/room noise, same as every other
    threshold in this project. If it's cutting you off mid-sentence,
    raise vad_silence_ms. If it's slow to start recording or misses
    quiet speech, lower vad_aggressiveness. If background noise keeps
    triggering it, raise vad_aggressiveness.

    Returns:
        recording_duration
    """
    vad_settings = settings["audio"]
    frame_ms = vad_settings["vad_frame_ms"]
    if frame_ms not in (10, 20, 30):
        raise ValueError(
            f"vad_frame_ms must be 10, 20, or 30 (webrtcvad's own "
            f"requirement) — got {frame_ms}"
        )
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)

    preroll_frames = int(vad_settings["vad_preroll_ms"] / frame_ms)
    silence_frames_needed = int(vad_settings["vad_silence_ms"] / frame_ms)
    min_speech_frames = int(vad_settings["vad_min_speech_ms"] / frame_ms)
    max_frames = int(vad_settings["vad_max_utterance_ms"] / frame_ms)

    vad = webrtcvad.Vad(vad_settings["vad_aggressiveness"])

    print("\n🎤 Listening... (speak whenever you're ready)")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=frame_samples,
    ) as stream:
        while True:
            preroll = deque(maxlen=preroll_frames)
            captured = []
            triggered = False
            silence_run = 0
            speech_frame_count = 0
            start = None

            while True:
                frame, overflowed = stream.read(frame_samples)
                if overflowed:
                    print("[MIC] input overflow")

                frame_bytes = bytes(frame)
                is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)

                if not triggered:
                    preroll.append(frame_bytes)
                    if is_speech:
                        triggered = True
                        start = time.perf_counter()
                        captured.extend(preroll)
                        speech_frame_count = 1
                        silence_run = 0
                    continue

                captured.append(frame_bytes)

                if is_speech:
                    speech_frame_count += 1
                    silence_run = 0
                else:
                    silence_run += 1

                if silence_run >= silence_frames_needed:
                    break
                if len(captured) >= max_frames:
                    print("[MIC] Hit max utterance length, cutting off.")
                    break

            if speech_frame_count >= min_speech_frames:
                audio_array = np.frombuffer(b"".join(captured), dtype=np.int16)
                raw_rms = (
                    float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)))
                    if audio_array.size else 0.0
                )
                if raw_rms >= vad_settings["vad_min_rms"]:
                    break
                print(
                    f"[MIC] RMS too low ({raw_rms:.0f}), likely noise "
                    f"not speech — listening again..."
                )
                continue

            # Too short to be real speech (cough, mic bump, etc). Discard
            # and listen again rather than sending noise to STT.
            print("[MIC] Too short, ignoring — listening again...")

    duration = time.perf_counter() - start
    _finalize_and_write(captured, output_path)
    print(f"[MIC] Recorded: {duration:.2f}s")

    return duration