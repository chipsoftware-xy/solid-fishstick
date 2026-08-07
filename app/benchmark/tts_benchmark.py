from app.tts.piper import TTS


tts = TTS()

elapsed = tts.speak(
    "Hello, this is your Smart Chip AI assistant.",
    "audio/output/response.wav",
)

print("\n===== RESULTS =====")
print(f"TTS Time : {elapsed:.2f}s")
print("Saved to : audio/output/response.wav")