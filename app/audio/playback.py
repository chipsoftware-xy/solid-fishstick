"""
Blocking WAV playback. Windows-only for now (uses the stdlib `winsound`
module, so no extra dependency), matching the current deployment target.

If you later need macOS/Linux support, swap the implementation below for
a small cross-platform library (e.g. `simpleaudio` or `sounddevice` +
`soundfile`) — voice_session.py only depends on the play_wav(path)
signature, so the swap is contained entirely to this file.
"""
import sys

if sys.platform == "win32":
    import winsound

    def play_wav(path: str):
        """Blocks until playback finishes."""
        winsound.PlaySound(path, winsound.SND_FILENAME)

else:
    def play_wav(path: str):
        raise NotImplementedError(
            "play_wav() only supports Windows right now (winsound). "
            "Install a cross-platform playback library (e.g. simpleaudio "
            "or sounddevice) and swap the implementation in this file."
        )