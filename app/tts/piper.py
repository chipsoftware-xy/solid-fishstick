from pathlib import Path
import time
import wave

from piper.voice import PiperVoice


class TTS:
    def __init__(self):
        self.model_path = Path("models/piper/voices/en_US-amy-medium.onnx")

        print("Loading Piper voice...")
        start = time.perf_counter()

        self.voice = PiperVoice.load(str(self.model_path))

        end = time.perf_counter()
        print(f"Piper loaded in {end - start:.2f}s")

    def speak(self, text: str, output_path: str):
        start = time.perf_counter()

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