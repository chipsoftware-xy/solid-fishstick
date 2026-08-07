import time
from faster_whisper import WhisperModel


class STT:
    def __init__(self, model_size="small"):
        print(f"Loading Whisper model ({model_size})...")
        start = time.perf_counter()

        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

        end = time.perf_counter()
        print(f"Model loaded in {end - start:.2f}s")

    def transcribe(self, audio_path):
        start = time.perf_counter()

        segments, info = self.model.transcribe(audio_path)

        text = " ".join(segment.text.strip() for segment in segments)

        end = time.perf_counter()

        return {
            "text": text,
            "language": info.language,
            "duration": end - start,
        }