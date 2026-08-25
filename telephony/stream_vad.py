"""
Turn detection for streamed call audio.

This is the phone-call equivalent of recording.py's
record_on_voice_activity() — same underlying idea (webrtcvad, a
pre-speech preroll buffer, a silence timeout that marks a turn as
"done"), but restructured to be PUSH-based rather than PULL-based:
recording.py blocks on a live mic stream and pulls frames in a loop;
this instead has audio pushed into it in small chunks as Exotel's
`media` events arrive over the WebSocket, and reports back whenever a
complete turn has been detected.

The VAD settings (aggressiveness, preroll, silence threshold, etc.)
were originally tuned against local mic input — real phone audio has
different noise/compression characteristics, so treat these as a
starting point to re-tune once tested against real calls, same as
every other VAD number in this project.
"""
import base64
from collections import deque

import numpy as np


def _compute_rms(pcm16_bytes):
    if not pcm16_bytes:
        return 0.0
    samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float64)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


class CallTurnDetector:
    def __init__(self, sample_rate=8000, settings=None):
        settings = settings or {}
        self.sample_rate = sample_rate
        self.frame_ms = settings.get("vad_frame_ms", 30)
        # 16-bit PCM = 2 bytes per sample.
        self.frame_bytes = int(sample_rate * self.frame_ms / 1000) * 2

        import webrtcvad
        self.vad = webrtcvad.Vad(settings.get("vad_aggressiveness", 2))

        preroll_frames = max(1, int(settings.get("vad_preroll_ms", 300) / self.frame_ms))
        self.preroll_buffer = deque(maxlen=preroll_frames)

        self.silence_frames_needed = max(1, int(settings.get("vad_silence_ms", 800) / self.frame_ms))
        self.min_speech_frames = max(1, int(settings.get("vad_min_speech_ms", 300) / self.frame_ms))
        self.max_utterance_frames = max(1, int(settings.get("vad_max_utterance_ms", 20000) / self.frame_ms))

        # Raw energy floor — VAD's per-frame "is this speech" classifier
        # can false-positive on quiet line noise/static, same as it did
        # for local mic input. Without this, that noise-classified-as-
        # speech gets sent straight to Whisper, which sometimes fills
        # audio with no real signal with hallucinated text (including,
        # confusingly, text resembling its own initial_prompt) rather
        # than admitting there was nothing there. Untested starting
        # point, same as every other VAD number here — watch for real
        # speech being wrongly discarded (raise it) vs hallucinated
        # transcripts still appearing (lower it further).
        self.min_rms = settings.get("vad_min_rms", 150)

        # Chunks arriving over the WebSocket won't line up neatly with
        # VAD frame boundaries — this holds whatever's left over
        # between calls to add_media_payload() until there's enough
        # for another full frame.
        self._leftover = b""
        self._reset_state()

    def _reset_state(self):
        self.triggered = False
        self.speech_frames = []
        self.silence_run = 0
        self.speech_frame_count = 0

    def add_media_payload(self, base64_payload):
        """
        Call this for every incoming Exotel 'media' event's payload.
        Returns the completed turn's raw PCM bytes once a full
        utterance has been detected (speech, then enough silence to
        count as "done talking"), or None if still listening / mid-
        utterance / the utterance was too short to count.
        """
        raw = base64.b64decode(base64_payload)
        self._leftover += raw

        turn_audio = None
        while len(self._leftover) >= self.frame_bytes:
            frame = self._leftover[:self.frame_bytes]
            self._leftover = self._leftover[self.frame_bytes:]
            result = self._process_frame(frame)
            if result is not None:
                turn_audio = result
        return turn_audio

    def _process_frame(self, frame):
        is_speech = self.vad.is_speech(frame, self.sample_rate)

        if not self.triggered:
            self.preroll_buffer.append(frame)
            if is_speech:
                self.triggered = True
                self.speech_frames = list(self.preroll_buffer)
                self.speech_frame_count = 1
                self.silence_run = 0
            return None

        self.speech_frames.append(frame)
        self.speech_frame_count += 1

        if is_speech:
            self.silence_run = 0
        else:
            self.silence_run += 1

        turn_finished = (
            self.silence_run >= self.silence_frames_needed
            or self.speech_frame_count >= self.max_utterance_frames
        )
        if not turn_finished:
            return None

        collected = self.speech_frames
        collected_frame_count = self.speech_frame_count
        self._reset_state()

        if collected_frame_count < self.min_speech_frames:
            # Too short to be real speech — discard, keep listening.
            return None

        audio_bytes = b"".join(collected)
        if _compute_rms(audio_bytes) < self.min_rms:
            # Long enough in frame-count, but too quiet to be real
            # speech — VAD likely false-triggered on line noise. Discard
            # rather than risk Whisper hallucinating on it.
            return None

        return audio_bytes