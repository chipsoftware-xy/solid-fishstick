import time
from faster_whisper import WhisperModel

# Biases decoding toward this vocabulary without any extra latency cost.
# Whisper's initial_prompt works best as a natural sentence resembling
# real transcript context, not a bare keyword list — this is what's
# behind mis-hearings like "HID Fargo" -> "top-high" or "DataCard" ->
# "data card interference": these are rare/brand terms the base model
# has little reason to prefer over a more common-sounding phrase unless
# nudged toward them. Keep this in sync with your actual product
# vocabulary (brands, card types, etc.) as it changes.
DOMAIN_VOCAB_PROMPT = (
    "Smart Chip Tree provides PAN India service across states including "
    "Arunachal Pradesh, Assam, Bihar, Delhi, Gujarat, Karnataka, Kerala, "
    "Maharashtra, Punjab, Rajasthan, Tamil Nadu, Telangana, Uttar Pradesh, "
    "and West Bengal. We provide PVC cards, MIFARE and RFID cards, ID card "
    "printers, and printer ribbons from Zebra, Evolis, HID Fargo, "
    "DataCard, and IDP retransfer printers, along with lanyards, access "
    "control, biometric attendance, and wholesale rates."
)


class STT:
    def __init__(self, model_size="tiny"):
        print(f"Loading Whisper model ({model_size})...")
        start = time.perf_counter()

        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

        end = time.perf_counter()
        print(f"Model loaded in {end - start:.2f}s")

    def transcribe(self, audio_path):
        start = time.perf_counter()

        segments, info = self.model.transcribe(
            audio_path,
            language="en",
            beam_size=1,
            initial_prompt=DOMAIN_VOCAB_PROMPT,
        )

        segments = list(segments)

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        # Average log-probability across segments as a rough confidence
        # signal — useful if you want chat_session to ask "could you
        # repeat that?" instead of confidently responding to a likely
        # mis-hearing. Not wired up yet; exposed here so it's available
        # if/when you want it.
        avg_logprob = None
        if segments:
            avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)

        end = time.perf_counter()

        return {
            "text": text,
            "language": info.language,
            "duration": end - start,
            "avg_logprob": avg_logprob,
        }