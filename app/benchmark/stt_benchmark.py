from app.stt.whisper import STT

stt = STT("small")

result = stt.transcribe("audio/input/test.wav")

print("\n===== RESULTS =====")
print(f"Language : {result['language']}")
print(f"Time     : {result['duration']:.2f}s")
print(f"Text     : {result['text']}")