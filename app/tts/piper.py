from pathlib import Path
import re
import time
import wave

from piper.voice import PiperVoice

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


def _replace_preserving_case(pattern: str, replacement: str, text: str) -> str:
    """Word-boundary regex replace that keeps the matched word's
    capitalization — "Great." -> "Wonderful.", "great" -> "wonderful"."""
    def repl(m):
        matched = m.group(0)
        if matched[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def _prepare_text(text: str) -> str:
    # Improve pronunciation for TTS without changing the actual transcript.
    text = text.replace("RFID", "R F I D")
    # Same category as RFID — this voice model reads "AI" as a single
    # slurred "ayi" sound instead of two distinct letters. Word-boundary
    # matched so it doesn't touch "AI" appearing inside another word.
    text = re.sub(r"\bAI\b", "A I", text)
    # Opposite problem from RFID: "PAN" here is a real word ("pan-India"
    # = across India), not an acronym — but Piper/espeak reads short
    # all-caps tokens as spelled-out letters by default ("P A N India").
    # Normalizing the case stops it from being treated as an acronym.
    text = re.sub(r"\bPAN\s+India\b", "Pan India", text, flags=re.IGNORECASE)
    # espeak reads "MI-" as a short-i prefix (like "mirror") instead of
    # the long-I "My" sound MIFARE actually uses. Splitting it so "My"
    # reads as the standalone word pushes it toward the right sound.
    text = re.sub(r"\bmifare\b", "My-fare", text, flags=re.IGNORECASE)
    # This specific voice model consistently mispronounces "great" —
    # tried "wonderful" first, but that also came out wrong (likely the
    # multi-syllable schwa+r combination). "good" is shorter and far
    # more phonetically simple, lower risk of the same problem — but
    # this is a best guess without being able to hear the actual
    # output; may need further adjustment.
    text = _replace_preserving_case(r"\bgreat\b", "good", text)
    # Same category of issue reported for "okay" — replacing with
    # "alright" as a first attempt for the same reason.
    text = _replace_preserving_case(r"\bokay\b", "alright", text)
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

    def synthesize_to_pcm(self, text: str):
        """
        Same synthesis as speak(), but returns raw PCM16 bytes and the
        voice's native sample rate directly, instead of writing a WAV
        file — for streamed call audio, which needs to be resampled to
        Exotel's rate and chunked for sending over a WebSocket rather
        than played locally.
        """
        text = clean_for_tts(text)
        text = _prepare_text(text)

        chunks = list(self.voice.synthesize(text))
        if not chunks:
            raise RuntimeError("No audio generated.")

        pcm_bytes = b"".join(chunk.audio_int16_bytes for chunk in chunks)
        sample_rate = chunks[0].sample_rate
        return pcm_bytes, sample_rate