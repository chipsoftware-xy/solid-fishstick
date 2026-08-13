from app.conversation.voice_session import VoiceSession


voice = VoiceSession()

# Simulate the customer already hearing Spark's opening.
voice.chat.send("hello")

# Now process the actual customer question, streaming sentence-by-
# sentence to TTS + playback instead of waiting for the full reply.
result = voice.process_audio_streaming(
    "audio/input/test2.wav"
)

if result:
    print("\n===== RESULT =====")
    print("User :", result["user_text"])
    print("Spark:", result["reply"])
    print("Chunks:", len(result["chunks"]))
    for i, chunk in enumerate(result["chunks"]):
        print(f"  [{i}] {chunk['audio_path']} :: {chunk['text']!r}")