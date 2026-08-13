from pathlib import Path
import time

from app.llm.chat_session import ChatSession
from app.stt.whisper import STT
from app.tts.piper import TTS
from app.audio.playback import play_wav


class VoiceSession:

    def __init__(self):
        print("Loading voice system...")

        self.stt = STT("tiny")
        self.tts = TTS()
        self.chat = ChatSession()

        print("Voice system ready.")

    def process_audio(self, audio_path):
        """
        Original blocking path: waits for the full LLM reply, then
        synthesizes it as one chunk. Left as-is for benchmarking /
        comparison against process_audio_streaming below.
        """
        stt_result = self.stt.transcribe(audio_path)
        user_text = stt_result["text"].strip()

        print(f"\nYou: {user_text}")

        if not user_text:
            return None

        reply = self.chat.send(user_text)

        print(f"Spark: {reply}")

        output_path = Path("audio/output/response.wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tts_time = self.tts.speak(reply, str(output_path))

        print(f"[VOICE] STT: {stt_result['duration']:.2f}s")
        print(f"[VOICE] TTS: {tts_time:.2f}s")

        return {
            "user_text": user_text,
            "reply": reply,
            "audio_path": str(output_path),
        }

    def process_audio_streaming(self, audio_path):
        """
        Streaming path for live calls. Each sentence is synthesized (and
        spoken, if TTS.speak() plays audio itself) as soon as it's ready,
        instead of waiting for the LLM to finish the whole reply first.
        The caller hears the first sentence within a couple of seconds
        rather than sitting through the full generation time in silence.

        Returns the same shape as process_audio(), plus a `chunks` list
        with per-sentence timing/paths if you need that level of detail
        (e.g. for debugging or a fancier playback queue later).
        """
        stt_start = time.perf_counter()
        stt_result = self.stt.transcribe(audio_path)
        user_text = stt_result["text"].strip()

        print(f"\nYou: {user_text}")

        if not user_text:
            return None

        output_dir = Path("audio/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        chunks = []
        turn_start = time.perf_counter()
        first_audio_at = None

        def on_sentence(sentence_text):
            nonlocal first_audio_at
            chunk_index = len(chunks)
            chunk_path = output_dir / f"response_{chunk_index}.wav"

            tts_time = self.tts.speak(sentence_text, str(chunk_path))

            if first_audio_at is None:
                first_audio_at = time.perf_counter()
                print(f"[VOICE] Time to first audio: {first_audio_at - turn_start:.2f}s")

            print(f"[VOICE] Sentence {chunk_index} TTS: {tts_time:.2f}s :: {sentence_text!r}")

            # Blocks until this sentence finishes playing. This is what
            # actually creates the overlap: Ollama keeps streaming tokens
            # for the NEXT sentence server-side the whole time this call
            # is blocked, so by the time playback finishes, some or all
            # of the next sentence is often already buffered and ready.
            play_start = time.perf_counter()
            play_wav(str(chunk_path))
            play_time = time.perf_counter() - play_start
            print(f"[VOICE] Sentence {chunk_index} playback: {play_time:.2f}s")

            chunks.append({
                "text": sentence_text,
                "audio_path": str(chunk_path),
                "tts_time": tts_time,
                "play_time": play_time,
            })

        reply = self.chat.send_streaming(user_text, on_sentence)

        print(f"Spark: {reply}")
        print(f"[VOICE] STT: {stt_result['duration']:.2f}s")
        print(f"[VOICE] Total turn: {time.perf_counter() - turn_start:.2f}s")

        return {
            "user_text": user_text,
            "reply": reply,
            "chunks": chunks,
            # First chunk kept under the same key as process_audio() for
            # any code that only expects a single audio_path.
            "audio_path": chunks[0]["audio_path"] if chunks else None,
        }