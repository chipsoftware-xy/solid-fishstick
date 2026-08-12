from app.conversation.voice_session import VoiceSession

voice = VoiceSession()

result = voice.process_audio(
    "audio/input/test.wav"
)

if result:
    print("\n===== RESULT =====")
    print("User :", result["user_text"])
    print("Spark:", result["reply"])
    print("Audio:", result["audio_path"])