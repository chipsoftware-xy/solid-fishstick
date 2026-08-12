from pathlib import Path

from app.llm.chat_session import ChatSession
from app.stt.whisper import STT
from app.tts.piper import TTS


class VoiceSession:

    def __init__(self):
        print("Loading voice system...")

        self.stt = STT("tiny")
        self.tts = TTS()
        self.chat = ChatSession()

        print("Voice system ready.")

    def process_audio(self, audio_path):
        # Speech -> text
        stt_result = self.stt.transcribe(audio_path)
        user_text = stt_result["text"].strip()

        print(f"\nYou: {user_text}")

        if not user_text:
            return None

        # Text -> Spark response
        reply = self.chat.send(user_text)

        print(f"Spark: {reply}")

        # Spark response -> speech
        output_path = Path("audio/output/response.wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tts_time = self.tts.speak(
            reply,
            str(output_path)
        )

        print(f"[VOICE] STT: {stt_result['duration']:.2f}s")
        print(f"[VOICE] TTS: {tts_time:.2f}s")

        return {
            "user_text": user_text,
            "reply": reply,
            "audio_path": str(output_path),
        }