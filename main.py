"""
Spark — AI sales caller. Main entry point.

Run with:
    python main.py
"""
from pathlib import Path

from app.conversation.voice_session import VoiceSession
from app.audio.recording import record_on_voice_activity
from app.call_finalizer import finalize_call


def main():
    voice = VoiceSession()

    print("\nSpark is listening. Just speak — no keypress needed.")
    print("The call ends automatically once wrapped up, or press Ctrl+C anytime.\n")

    audio_path = Path("audio/input/live_input.wav")

    try:
        while True:
            record_on_voice_activity(str(audio_path))

            result = voice.process_audio_streaming(str(audio_path))

            if not result:
                continue

            print("\n===== RESULT =====")
            print("You  :", result["user_text"])
            print("Spark:", result["reply"])

            if voice.chat.call_finished:
                print("\nCall wrapped up — ending automatically...")
                finalize_call(voice.chat)
                break

    except KeyboardInterrupt:
        print("\n\nEnding call...")
        if not voice.chat.call_finished:
            finalize_call(voice.chat)


if __name__ == "__main__":
    main()