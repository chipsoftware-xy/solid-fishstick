from pathlib import Path
import time

from app.llm.chat_session import ChatSession
from app.stt.whisper import STT
from app.tts.piper import TTS
from app.audio.playback import play_wav

# avg_logprob from faster_whisper: closer to 0 = confident, more
# negative = uncertain. Below this threshold, the transcription is
# treated as unreliable and Spark asks the customer to repeat themselves
# instead of handing possibly-garbled text to the LLM.
#
# IMPORTANT: this threshold is NOT portable across Whisper model sizes.
# It was tuned loosely against "tiny"; switching to "base" shifted the
# whole score distribution more negative — three CORRECT "Hello."
# transcriptions scored -0.98 to -1.12 with base, all comfortably below
# the old -0.7 cutoff, so all three were wrongly rejected. Short
# utterances also tend to score more negatively in general (less context
# for the model to be confident about), independent of model choice.
# -1.2 is a looser starting point for base — but treat this as still
# untested: log avg_logprob on a real mix of correct short utterances,
# correct long ones, and genuinely garbled audio, and set the cutoff
# just below where the correct ones cluster. Re-tune again if you change
# model size in the future.
LOW_CONFIDENCE_THRESHOLD = -1.2

CLARIFICATION_MESSAGE = "Sorry, I didn't quite catch that — could you say that again?"


class VoiceSession:

    def __init__(self):
        print("Loading voice system...")

        self.stt = STT("base")
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

        If the transcription is empty or low-confidence, this skips the
        LLM entirely and asks the customer to repeat themselves —
        avoiding a confident, fabricated response built on garbled input
        (this is a real failure mode, not theoretical: e.g. Whisper
        mis-hearing unclear audio as "PitchRack" and the LLM inventing a
        fake product integration to explain it).

        Returns the same shape as process_audio(), plus a `chunks` list
        with per-sentence timing/paths if you need that level of detail
        (e.g. for debugging or a fancier playback queue later).
        """
        stt_result = self.stt.transcribe(audio_path)
        user_text = stt_result["text"].strip()
        avg_logprob = stt_result.get("avg_logprob")

        print(f"\nYou: {user_text}")
        if avg_logprob is not None:
            print(f"[VOICE] STT confidence (avg_logprob): {avg_logprob:.3f}")

        low_confidence = (
            avg_logprob is not None and avg_logprob < LOW_CONFIDENCE_THRESHOLD
        )

        if not user_text or low_confidence:
            reason = "empty transcription" if not user_text else "low STT confidence"
            print(f"[VOICE] Skipping LLM ({reason}) — asking customer to repeat.")

            # Log to the same transcript as normal turns, even though this
            # bypasses chat_session's own dispatch, so the call record
            # stays complete for later review/extraction.
            if user_text:
                self.chat._log("user", user_text)
            self.chat._log("assistant", CLARIFICATION_MESSAGE)

            output_dir = Path("audio/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = output_dir / "response_0.wav"
            self.tts.speak(CLARIFICATION_MESSAGE, str(chunk_path))
            play_wav(str(chunk_path))

            return {
                "user_text": user_text,
                "reply": CLARIFICATION_MESSAGE,
                "chunks": [{"text": CLARIFICATION_MESSAGE, "audio_path": str(chunk_path)}],
                "audio_path": str(chunk_path),
                "low_confidence": low_confidence,
            }

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