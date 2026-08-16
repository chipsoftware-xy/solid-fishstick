"""
Transcribes an original recording and its phone-quality-simulated
counterpart side by side, so you can directly compare STT accuracy
between "clean mic audio" and "realistic phone call audio" without
needing a live call or telephony integration.

Usage:
    python compare_phone_stt.py path/to/original.wav
"""

import sys
from app.stt.whisper import STT
from stimulate_phone_call import simulate_phone_audio


def main(original_path):
    phone_path = simulate_phone_audio(original_path)

    stt = STT("base")  # match whatever model size you're actually using

    print("\nTranscribing original (clean mic) audio...")
    original_result = stt.transcribe(original_path)

    print("Transcribing phone-simulated audio...")
    phone_result = stt.transcribe(phone_path)

    print("\n===== COMPARISON =====")
    print(f"Original      : {original_result['text']!r}")
    print(f"  avg_logprob : {original_result['avg_logprob']:.3f}")
    print(f"Phone-simulated: {phone_result['text']!r}")
    print(f"  avg_logprob : {phone_result['avg_logprob']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_phone_stt.py <original.wav>")
        sys.exit(1)
    main(sys.argv[1])