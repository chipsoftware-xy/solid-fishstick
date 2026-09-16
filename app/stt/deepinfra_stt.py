"""
DeepInfra-hosted Whisper STT client — replaces telephony/dual_tier_stt.py's
local, dual-model (base/small) approach with a single, much more accurate
cloud model.

Built because a real, repeated pattern across tonight's test calls traced
back to STT accuracy, not LLM reasoning: "to-let" transcribed as "toilet",
a customer's actual organization "BKL" transcribed as "Began", a spoken
name garbled into "The Pan". The LLM was faithfully working with corrupted
input in every one of these cases — no amount of prompt tuning fixes a
transcript that's already wrong before the LLM ever sees it.

Exposes the same transcribe_array(samples, duration_s) shape that
DualTierSTT already provides, so two_way_test.py's call site only needs
its import/instantiation changed, not its calling code — EXCEPT that this
version is async (the call site currently calls it directly in the main
event-loop coroutine, not from a background thread the way the LLM call
is; a synchronous network request there would block ws.recv() for the
STT's full round-trip time, which the local model already did for up to
5+ seconds per real logs tonight — worth fixing regardless of this
switch, and this async version does that rather than reproducing it).

DeepInfra exposes Whisper through the same OpenAI-compatible endpoint
already used for the LLM (https://api.deepinfra.com/v1/openai), via
audio.transcriptions.create() instead of chat.completions.create() -
confirmed directly against DeepInfra's own documentation
(https://docs.deepinfra.com/tutorials/whisper) before writing this. Uses
the official `openai` package's AsyncOpenAI client, consistent with how
deepinfra_client.py (the LLM client) already uses the sync OpenAI client.

Requires: pip install openai --break-system-packages
"""
import io
import time
import wave

import numpy as np
from openai import AsyncOpenAI

from app.config import settings

_STT_SETTINGS = settings["stt"]
_DEEPINFRA_SETTINGS = _STT_SETTINGS["deepinfra"]

# No longer passed to the transcription call below 
DOMAIN_VOCAB_PROMPT = (
    "Smart Chip Tree provides PAN India service across states including "
    "Arunachal Pradesh, Assam, Bihar, Delhi, Gujarat, Karnataka, Kerala, "
    "Maharashtra, Punjab, Rajasthan, Tamil Nadu, Telangana, Uttar Pradesh, "
    "and West Bengal. We provide PVC cards, MIFARE and RFID cards, ID card "
    "printers, and printer ribbons from Zebra, Evolis, HID Fargo, "
    "DataCard, and IDP retransfer printers, along with lanyards, access "
    "control, biometric attendance, and wholesale rates."
)

# Used ONLY during actual name collection
NAME_HINT_PROMPT = (
    "The customer's name may be Ishita, Priya, Rohit, Amit, Sneha, "
    "Arjun, Kavya, Vikram, Anjali, Rahul, Pooja, Suresh, Divya, Karan, "
    "Meera, or a similar Indian name."
)


def _samples_to_wav_bytes(samples, sample_rate):
   
    int16_samples = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16_samples.tobytes())

    buffer.seek(0)
  
    buffer.name = "audio.wav"
    return buffer


class STT:
    def __init__(self, model=None, sample_rate=16000):
        api_key = _DEEPINFRA_SETTINGS["api_key"]
        if not api_key:
            raise ValueError(
                "deepinfra_stt.STT is being used, but stt.deepinfra.api_key "
                "in settings.yml is empty. Set it to your DeepInfra API key "
                "before running — this can be the same key as "
                "llm.deepinfra.api_key if you're using DeepInfra for both."
            )

        self.model = model or _DEEPINFRA_SETTINGS["model"]
        self.sample_rate = sample_rate
        
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai", timeout=12.0)

    async def verify(self):
     
        print(f"Verifying DeepInfra access to {self.model}...")
        silence = np.zeros(self.sample_rate // 2, dtype=np.float32)  # 0.5s
        await self.transcribe_array(silence, duration_s=0.5)
        print("STT ready.")

    async def transcribe_array(self, samples, duration_s, prompt=None, allow_other_languages=False, service_tier=None):
       
        start = time.perf_counter()

        wav_buffer = _samples_to_wav_bytes(samples, self.sample_rate)
        avg_logprob = None
        no_speech_prob = None
        compression_ratio = None

        try:
            create_kwargs = {
                "model": self.model,
                "file": wav_buffer,
                "response_format": "verbose_json",
                "temperature": 0.0, 
            }
            if service_tier:
                # DeepInfra-specific field
                create_kwargs["extra_body"] = {"service_tier": service_tier}
            if not allow_other_languages:
                create_kwargs["language"] = "en"
            if prompt:
                create_kwargs["prompt"] = prompt
            response = await self.client.audio.transcriptions.create(**create_kwargs)
            text = response.text.strip()
            segments = getattr(response, "segments", None) or []
            avg_logprob = min((s.avg_logprob for s in segments), default=None)
            no_speech_prob = max((s.no_speech_prob for s in segments), default=None)
            compression_ratio = max((s.compression_ratio for s in segments), default=None)
        except Exception as e:
            
            print(f"[WARN] DeepInfra STT call failed, treating as empty transcript: {e}")
            text = ""

        end = time.perf_counter()

        return {
            "text": text,
            "language": "en",
            "duration": end - start,
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
            "compression_ratio": compression_ratio,
            "model_used": self.model,
        }