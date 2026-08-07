from app.stt.whisper import STT
from app.llm.ollama_client import LLM
from app.tts.piper import TTS
from app.utils.text_cleaner import clean_for_tts

print("Loading models...\n")

stt = STT()
llm = LLM()
tts = TTS()

print("\n========== PIPELINE ==========\n")

stt_result = stt.transcribe("audio/input/test.wav")

llm_result = llm.generate(stt_result["text"])
speech=clean_for_tts(llm_result["text"])
tts_time = tts.speak(
    speech,
    "audio/output/response.wav"
)

total = (
    stt_result["duration"]
    + llm_result["duration"]
    + tts_time
)

print("Transcript:")
print(stt_result["text"])

print("\nReply:")
print(llm_result["text"])

print("\n========== TIMINGS ==========")
print(f"STT : {stt_result['duration']:.2f}s")
print(f"LLM : {llm_result['duration']:.2f}s")
print(f"TTS : {tts_time:.2f}s")
print("----------------------------")
print(f"TOTAL : {total:.2f}s")

print("\nSaved:")
print("audio/output/response.wav")