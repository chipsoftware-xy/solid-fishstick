from pathlib import Path

from app.audio.recording import record_until_enter
from app.conversation.voice_session import VoiceSession


voice = VoiceSession()

print("\nSpark is listening. Speak when ready.")

while True:

    audio_path = Path("audio/input/live_input.wav")
    duration, had_speech = record_until_enter(str(audio_path))
    if not had_speech:
        print("Spark: (heard nothing, waiting for you to speak.)")
        continue

    result = voice.process_audio_streaming(str(audio_path))

    if not result:
        continue

    print("\n===== RESULT =====")
    print("You  :", result["user_text"])
    print("Spark:", result["reply"])

    answer = input("\nPress ENTER for another turn, or type 'exit': ")

    if answer.strip().lower() == "exit":
        break