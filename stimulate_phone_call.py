"""
Degrades a WAV recording to approximate real phone-call audio quality —
bandpass filtering, 8kHz downsampling, and mu-law (G.711) compression —
so you can test STT accuracy against realistic call conditions without
a telephony integration set up yet.

Usage:
    python simulate_phone_audio.py path/to/input.wav [path/to/output.wav]

If no output path is given, writes alongside the input as
"<name>_phone_sim.wav".

Requires scipy (pip install scipy --break-system-packages if not
already installed — numpy is likely already present given your
recording pipeline).
"""

import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, resample_poly

TELEPHONE_LOW_HZ = 300
TELEPHONE_HIGH_HZ = 3400
PHONE_SAMPLE_RATE = 8000
TARGET_SAMPLE_RATE = 16000  # what the rest of your pipeline expects


def _read_wav(path):
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        n_frames = wav.getnframes()
        data = wav.readframes(n_frames)
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return audio, rate


def _write_wav(path, audio, rate):
    audio_int16 = np.clip(audio, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(audio_int16.tobytes())


def _bandpass(audio, rate, low_hz=TELEPHONE_LOW_HZ, high_hz=TELEPHONE_HIGH_HZ):
    nyquist = rate / 2
    sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")
    return sosfilt(sos, audio)


def _mu_law_round_trip(audio, mu=255):
    """
    Encodes then immediately decodes through mu-law compression, the
    codec (G.711) most phone calls still actually use, to add the same
    quantization artifacts real call audio would have.
    """
    x = np.clip(audio / 32768.0, -1.0, 1.0)
    compressed = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.round(compressed * 127) / 127  # 8-bit mu-law levels
    decoded = np.sign(quantized) * (1.0 / mu) * (np.power(1 + mu, np.abs(quantized)) - 1)
    return decoded * 32768.0


def simulate_phone_audio(input_path, output_path=None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_phone_sim.wav")

    audio, rate = _read_wav(input_path)

    # 1. Telephone bandwidth — real lines cut everything outside 300-3400Hz.
    audio = _bandpass(audio, rate)

    # 2. Down to 8kHz and back up — genuine resolution loss, not just a
    # filter. Whisper (and your pipeline) expects 16kHz input, so we
    # upsample back rather than feeding raw 8kHz.
    audio_8k = resample_poly(audio, PHONE_SAMPLE_RATE, rate)
    audio_16k = resample_poly(audio_8k, TARGET_SAMPLE_RATE, PHONE_SAMPLE_RATE)

    # 3. Codec compression artifacts.
    audio_16k = _mu_law_round_trip(audio_16k)

    _write_wav(output_path, audio_16k, TARGET_SAMPLE_RATE)
    print(f"Simulated phone-quality audio written to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simulate_phone_audio.py <input.wav> [output.wav]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    simulate_phone_audio(in_path, out_path)