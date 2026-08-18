"""
The actual "run one call" logic, extracted out of main.py so it has a
single home — both the CLI entry point and the web UI's Start button
call this same function, rather than duplicating the loop in two places.
"""
from pathlib import Path

from app.conversation.voice_session import VoiceSession
from app.audio.recording import record_on_voice_activity
from app.call_finalizer import finalize_call


def run_call(voice=None, stop_event=None):
    """
    Runs one full call: loops listening -> processing until the call
    auto-ends, is interrupted (Ctrl+C, when run from a terminal), or a
    stop is requested, then finalizes it.

    `voice`, if given, is an EXISTING VoiceSession to reuse — this is
    what lets repeated calls within the same running process (e.g. the
    web UI's Start button, clicked again after a previous call ends)
    skip reloading Whisper/Piper/the LLM/knowledge base every time.
    Only ChatSession (the actual per-call conversation state) gets
    reset via start_new_call(). If not given, a fresh VoiceSession is
    created (matches the old behavior — fine for a single one-off
    call, e.g. from the CLI, which only ever runs one call per process
    anyway).

    `stop_event` (a threading.Event, used by the web UI's Stop button)
    is checked between turns — a call can't be interrupted mid-recording
    or mid-generation without much more invasive threading, so stopping
    takes effect as soon as the current exchange finishes, not
    instantly. This is what actually makes stopping possible for a
    web-triggered call in the first place: Ctrl+C only reaches Python's
    main thread, not a background thread a call is running in, so it
    can't be used to stop a call started from the web UI at all.
    """
    if voice is None:
        voice = VoiceSession()
    else:
        voice.start_new_call()

    print("\nSpark is listening. Just speak — no keypress needed.")
    print("The call ends automatically once wrapped up, or press Ctrl+C anytime.\n")

    audio_path = Path("audio/input/live_input.wav")
    stopped_early = False

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                stopped_early = True
                break

            record_on_voice_activity(str(audio_path))

            result = voice.process_audio_streaming(str(audio_path))

            if not result:
                continue

            print("\n===== RESULT =====")
            print("You  :", result["user_text"])
            print("Spark:", result["reply"])

            if voice.chat.call_finished:
                print("\nCall wrapped up — ending automatically...")
                break

    except KeyboardInterrupt:
        print("\n\nEnding call...")
        stopped_early = True

    if stopped_early:
        print("Call stopped before finishing normally.")

    finalize_call(voice.chat)