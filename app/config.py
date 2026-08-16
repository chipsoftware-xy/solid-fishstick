"""
Central settings loader. Reads config/settings.yml and merges it over a
set of hardcoded defaults, so a missing file or a missing individual key
never crashes anything — it just falls back silently.

Usage elsewhere in the app:
    from app.config import settings
    settings["llm"]["model"]
    settings["stt"]["low_confidence_threshold"]
"""
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yml"

# These are the values we've actually landed on through testing in this
# project — treat settings.yml as the place to override them, not as the
# only place they're documented. If settings.yml is missing or a key is
# absent, these are what's actually used.
_DEFAULTS = {
    "llm": {
        "provider": "ollama",
        "model": "gemma3:4b",
        "temperature": 0.3,
        # ~35 words is roughly 45-55 tokens for English; capped a
        # little above that as a hard backstop on reply length.
        "num_predict": 55,
        # Matches typical prompt size (system prompt + knowledge +
        # trimmed history) plus headroom for the reply.
        "num_ctx": 3072,
        # -1 keeps the model loaded indefinitely between calls, so a
        # new call doesn't pay a reload penalty.
        "keep_alive": -1,
    },
    "stt": {
        "model": "base",
        # avg_logprob threshold below which a transcription is treated
        # as unreliable and Spark asks the customer to repeat
        # themselves. Model-dependent — re-tune if you change stt.model.
        "low_confidence_threshold": -1.2,
    },
    "tts": {
        "voice": "en_US-lessac-high",
    },
    "audio": {
        "sample_rate": 16000,
        # Target peak amplitude after gain normalization (int16 max is
        # 32767) — leaves headroom to avoid clipping.
        "target_peak": 26000,
        # Cap on how much a quiet recording gets amplified, so a near-
        # silent clip doesn't get boosted into pure noise.
        "max_gain": 8.0,
        # Silence padding added to both ends of a recording, so speech
        # right at the start/end of the clip doesn't get clipped mid-word.
        "silence_pad_ms": 300,
        # Voice activity detection — automatic start/stop of recording
        # without a keypress. All of these are untested starting points;
        # expect to re-tune against your actual mic/room noise, the same
        # way the confidence thresholds elsewhere needed real-world
        # calibration.
        "vad_aggressiveness": 2,  # 0 (permissive) to 3 (strict) — webrtcvad's own scale
        "vad_frame_ms": 30,  # webrtcvad only accepts 10, 20, or 30
        "vad_preroll_ms": 300,  # buffered before trigger, so speech onset isn't clipped
        "vad_silence_ms": 800,  # sustained silence required to end the utterance
        "vad_min_speech_ms": 300,  # discard shorter blips (coughs, mic bumps) as noise
        # Raw (pre-gain) RMS floor. VAD can misclassify near-silent room
        # noise as "speech" (frame-level detection, not amplitude-aware)
        # — Whisper then does its best to produce SOMETHING plausible
        # from that near-silent audio, which can come out as bizarre
        # hallucinated text unrelated to anything actually said. This
        # catches that case even when VAD and the duration check both
        # let it through. Untested starting point like the others here.
        "vad_min_rms": 150,
        "vad_max_utterance_ms": 20000,  # safety cap so it can't record forever
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return _deep_merge(_DEFAULTS, loaded)
    return dict(_DEFAULTS)


settings = _load()