import time

from elevenlabs.client import ElevenLabs

from app.config import settings

_TTS_SETTINGS = settings["tts"]
_ELEVENLABS_SETTINGS = _TTS_SETTINGS["elevenlabs"]

# eleven_turbo_v2_5 specifically, not eleven_v3 — v3 trades speed for
# maximum expressiveness (a demo/quality-showcase model), while turbo
# is ElevenLabs' own recommended choice for real-time conversational
# use where latency matters as much as quality.
_MODEL_ID = "eleven_turbo_v2_5"

# pcm_16000 requested directly — confirmed via ElevenLabs' own API
# reference as a valid, unrestricted output format (no tier
# requirement, unlike pcm_44100/wav_44100 which need Pro tier). This
# means no MP3 decoding step at all: what comes back is already raw,
# 16-bit PCM at 16kHz, matching what the rest of this pipeline already
# expects (WHISPER_SAMPLE_RATE), so this can resample down to whatever
# call_sample_rate actually is the same way Piper's output already does.
_OUTPUT_FORMAT = "pcm_16000"
NATIVE_SAMPLE_RATE = 16000


class TTS:
    def __init__(self, voice_id=None):
        api_key = _ELEVENLABS_SETTINGS["api_key"]
        if not api_key:
            raise ValueError(
                "elevenlabs_tts.TTS is being used, but tts.elevenlabs.api_key "
                "in settings.yml is empty. Set it to your ElevenLabs API key "
                "before running."
            )
        self.voice_id = voice_id or _ELEVENLABS_SETTINGS["voice_id"]
        if not self.voice_id:
            raise ValueError(
                "elevenlabs_tts.TTS is being used, but tts.elevenlabs.voice_id "
                "in settings.yml is empty. Multiple different voices in "
                "ElevenLabs' library share the same display name (confirmed "
                "directly — several unrelated voices are all named \"Anvi\", "
                "with wildly different tones), so the exact voice_id — not "
                "the name — has to be copied from the voice's own page in "
                "the ElevenLabs dashboard and set here."
            )
        self.client = ElevenLabs(api_key=api_key)

    def synthesize_to_pcm(self, text):
        """
        Same call shape as piper.py's TTS.synthesize_to_pcm() — returns
        (pcm_bytes, native_sample_rate) — so this is a drop-in
        replacement; two_way_test.py needs no changes beyond the import
        line.
        """
        start = time.perf_counter()
        chunks = self.client.text_to_speech.stream(
            voice_id=self.voice_id,
            text=text,
            model_id=_MODEL_ID,
            output_format=_OUTPUT_FORMAT,
        )
        # convert() returns Iterator[bytes], not a single bytes object —
        # confirmed directly via the real SDK's method signature, not
        # assumed — so the streamed chunks need concatenating into one
        # buffer before this can be treated as a complete PCM clip.
        pcm_bytes = b"".join(chunks)
        elapsed = time.perf_counter() - start
        print(f"[TTS] ElevenLabs synth: {elapsed:.2f}s for {len(text.split())} words")
        return pcm_bytes, NATIVE_SAMPLE_RATE