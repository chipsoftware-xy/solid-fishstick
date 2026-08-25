"""
Dual-tier speech-to-text for streamed call audio.

Real testing (isolated single words like "sure" spoken with silence on
both sides, repeated multiple times) showed the fast ("base") Whisper
model's accuracy craters specifically on SHORT, isolated utterances —
long natural sentences already transcribed excellently on "base" every
time. This makes sense: Whisper leans heavily on surrounding linguistic
context to disambiguate acoustically similar sounds, and a bare short
word gives it almost nothing to work with — worse on already-degraded
8kHz phone audio than it would be on a clean mic recording.

Duration is used as the trigger, NOT confidence score — this project
already proved Whisper can be confidently wrong (see the near-silent-
audio hallucination fix earlier), so confidence isn't a reliable signal
for this failure mode. Duration was the one thing that predicted the
failure with total consistency in testing.

This keeps the added latency cost contained: only short turns (which
are cheap to transcribe even with a slower model, since there's just
not much audio to process) pay for the more accurate model. Every
normal-or-longer turn stays exactly as fast as it always was.
"""
from app.stt.whisper import STT

SHORT_TURN_THRESHOLD_S = 2.0


class DualTierSTT:
    def __init__(self, fast_model="base", accurate_model="small"):
        self.fast = STT(model_size=fast_model)
        self.accurate = STT(model_size=accurate_model)

    def transcribe_array(self, samples, duration_s):
        if duration_s < SHORT_TURN_THRESHOLD_S:
            model, model_name = self.accurate, "small"
        else:
            model, model_name = self.fast, "base"

        result = model.transcribe_array(samples)
        result["model_used"] = model_name
        return result