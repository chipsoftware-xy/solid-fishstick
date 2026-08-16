from pathlib import Path
import time
import wave

from piper.voice import PiperVoice
import re

from app.config import settings
from app.utils.text_cleaner import clean_for_tts

# Default voice comes from config/settings.yml (tts.voice) — override by
# passing voice_name= explicitly to TTS() for a quick A/B test without
# touching settings.yml.
VOICES_DIR = Path("models/piper/voices")

# Each voice needs BOTH of these files, same base name, placed in
# VOICES_DIR:
#   {voice_name}.onnx
#   {voice_name}.onnx.json
#
# Download commands (run from your project root, adjust for your shell):
#
#   en_US-lessac-high (clearer, female, ~114MB) — current default:
#     curl -L -o models/piper/voices/en_US-lessac-high.onnx \
#       https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx
#     curl -L -o models/piper/voices/en_US-lessac-high.onnx.json \
#       https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx.json
#
#   en_US-hfc_male-medium (male, ~63MB):
#     curl -L -o models/piper/voices/en_US-hfc_male-medium.onnx \
#       https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx
#     curl -L -o models/piper/voices/en_US-hfc_male-medium.onnx.json \
#       https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx.json
#
# On Windows PowerShell, replace `curl -L -o` with:
#   Invoke-WebRequest -Uri <url> -OutFile <path>


def _prepare_text(text: str) -> str:
    # Improve pronunciation for TTS without changing the actual transcript.
    text = text.replace("RFID", "R F I D")
    text = re.sub(r"\bPAN\s+India\b", "Pan India", text, flags=re.IGNORECASE)
    return text


class TTS:
    def __init__(self, voice_name: str = None):
        voice_name = voice_name or settings["tts"]["voice"]
        self.model_path = VOICES_DIR / f"{voice_name}.onnx"
        config_path = VOICES_DIR / f"{voice_name}.onnx.json"

        if not self.model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Missing Piper voice files for '{voice_name}'. Expected "
                f"both:\n  {self.model_path}\n  {config_path}\n"
                f"See the download commands at the top of this file."
            )

        print(f"Loading Piper voice ({voice_name})...")
        start = time.perf_counter()

        self.voice = PiperVoice.load(str(self.model_path))

        end = time.perf_counter()
        print(f"Piper loaded in {end - start:.2f}s")

    def speak(self, text: str, output_path: str):
        start = time.perf_counter()

        # Only modify the text going into Piper.
        # The original Spark response remains unchanged in the transcript.
        # clean_for_tts strips markdown artifacts (**bold**, `code`,
        # bullets, headers) the model isn't supposed to produce per the
        # system prompt, but occasionally does anyway — without this,
        # Piper would literally speak stray asterisks/hashes out loud.
        text = clean_for_tts(text)
        text = _prepare_text(text)

        chunks = list(self.voice.synthesize(text))

        if not chunks:
            raise RuntimeError("No audio generated.")

        with wave.open(output_path, "wb") as wav:
            wav.setnchannels(chunks[0].sample_channels)
            wav.setsampwidth(chunks[0].sample_width)
            wav.setframerate(chunks[0].sample_rate)

            for chunk in chunks:
                wav.writeframes(chunk.audio_int16_bytes)

        end = time.perf_counter()

        return end - start