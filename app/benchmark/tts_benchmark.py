from app.conversation.voice_session import VoiceSession

voice = VoiceSession()

# Start the call so Spark's opening is already established.
voice.chat.send("hello")

print("\n========== TURN 1 ==========")

result1 = voice.process_audio_streaming(
    "audio/input/test2.wav"
)

print("\n========== TURN 2 ==========")

result2 = voice.process_audio_streaming(
    "audio/input/test3.wav"
)