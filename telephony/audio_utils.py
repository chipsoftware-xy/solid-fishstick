"""
Small audio conversion helpers for streamed call audio.
"""
import numpy as np


def resample_pcm16(pcm16_bytes, from_rate, to_rate):
    """
    Resamples raw 16-bit PCM audio from one sample rate to another using
    simple linear interpolation — not audiophile-grade, but perfectly
    adequate for speech recognition input, and avoids either a heavier
    resampling library or the stdlib `audioop` module (deprecated, and
    being removed from Python entirely).
    """
    if from_rate == to_rate:
        return np.frombuffer(pcm16_bytes, dtype=np.int16)

    samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
    n = len(samples)
    if n == 0:
        return samples.astype(np.int16)

    duration = n / from_rate
    n_new = int(duration * to_rate)
    x_old = np.linspace(0, duration, n, endpoint=False)
    x_new = np.linspace(0, duration, n_new, endpoint=False)
    resampled = np.interp(x_new, x_old, samples)
    return resampled.astype(np.int16)


def pcm16_to_whisper_input(pcm16_array):
    """Converts int16 PCM samples to the float32, [-1.0, 1.0]-range
    array Whisper expects."""
    return pcm16_array.astype(np.float32) / 32768.0