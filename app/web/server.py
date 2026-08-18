"""
SparkSCT web UI — a local-only Flask app. No auth, no production WSGI
server needed: each device runs its own instance against its own local
resources (Ollama, mic, DB), so this is meant to be reached only via
127.0.0.1 on the same machine.
"""
import json
import os
import sqlite3
import threading
import time

from flask import Flask, render_template, redirect, url_for, flash

from app.conversation.call_runner import run_call
from app.conversation.voice_session import VoiceSession
from app.call_storage import DB_PATH

app = Flask(__name__)
app.secret_key = "spark-local-dev"  # local-only app, no real auth surface

# Guards against starting a second call while one is already using the
# mic/speakers — both are single, shared local hardware resources. Also
# guards creation of the shared VoiceSession below, so two near-
# simultaneous Start clicks can't both try to build it at once.
_call_lock = threading.Lock()
_call_in_progress = False

# Signals a running call to wind down after its current exchange
# finishes. Cleared at the start of each new call so a stale "stop"
# from a previous call never carries over.
_stop_event = threading.Event()

# Built once, on the FIRST Start click — this is what actually loads
# Whisper/Piper/the LLM/knowledge base. Every call after that reuses
# this same instance (via VoiceSession.start_new_call(), which only
# resets the per-call ChatSession) instead of paying that reload cost
# again — closer to how a real batch of calls will eventually behave,
# where all of that only needs to load once per run, not once per call.
_voice_session = None


def _get_recent_calls(limit=5):
    """
    The most recent calls, most recent first, each with its transcript
    parsed from JSON into a list the template can loop over. Shared by
    the main page's "Last call" box (limit=1) and Review Conversations
    (limit=5), so there's one query instead of two near-duplicates.
    """
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM calls ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    calls = []
    for row in rows:
        call = dict(row)
        try:
            call["transcript"] = json.loads(call["transcript"] or "[]")
        except (json.JSONDecodeError, TypeError):
            call["transcript"] = []
        calls.append(call)
    return calls


def _get_latest_call():
    """
    The most recent call — this only exists because finalize_call() ran,
    which only happens when a call genuinely finished. Used as the "did
    a call actually end" confirmation on the page, grounded in real
    saved data rather than an in-memory flag.
    """
    calls = _get_recent_calls(limit=1)
    return calls[0] if calls else None


def _run_call_and_release():
    global _call_in_progress, _voice_session
    _stop_event.clear()
    try:
        if _voice_session is None:
            # First call in this process — this is what pays the real
            # loading cost (Whisper, Piper, LLM warmup, knowledge base).
            # Done here in the background thread, not in the request
            # handler, so the page responds instantly even on the very
            # first Start click instead of hanging on the load.
            _voice_session = VoiceSession()
        run_call(voice=_voice_session, stop_event=_stop_event)
    finally:
        with _call_lock:
            _call_in_progress = False


@app.route("/")
def index():
    with _call_lock:
        in_progress = _call_in_progress
    return render_template(
        "index.html",
        call_in_progress=in_progress,
        latest_call=_get_latest_call(),
    )


@app.route("/start-call", methods=["POST"])
def start_call():
    global _call_in_progress
    with _call_lock:
        if _call_in_progress:
            flash("A call is already in progress.")
            return redirect(url_for("index"))
        _call_in_progress = True
        first_load = _voice_session is None

    thread = threading.Thread(target=_run_call_and_release, daemon=True)
    thread.start()

    if first_load:
        flash("Loading voice system for the first time — this may take a few seconds. Watch your terminal.")
    else:
        flash("Call started — watch your terminal for live progress.")
    return redirect(url_for("index"))


@app.route("/stop-call", methods=["POST"])
def stop_call():
    with _call_lock:
        in_progress = _call_in_progress
    if not in_progress:
        flash("No call is currently in progress.")
        return redirect(url_for("index"))

    _stop_event.set()
    flash("Stopping — the current exchange will finish, then the call will end.")
    return redirect(url_for("index"))


@app.route("/stop-session", methods=["POST"])
def stop_session():
    """
    "Stop Gemma" — actually unloads the model from Ollama's memory and
    drops our cached VoiceSession, so the NEXT Start click does a full
    fresh reload again (Whisper, Piper, LLM warmup, knowledge base),
    same as the very first call. This is separate from stopping a
    call: a call ending leaves the model loaded (deliberately, so the
    next call is fast); this is the on-demand way to actually release
    it when you're done for a while.
    """
    global _voice_session
    with _call_lock:
        if _call_in_progress:
            flash("Stop the current call before stopping Gemma.")
            return redirect(url_for("index"))

        if _voice_session is None:
            flash("Nothing is loaded right now.")
            return redirect(url_for("index"))

        _voice_session._llm.unload()
        _voice_session = None

    flash("Gemma stopped. The next call will reload everything from scratch.")
    return redirect(url_for("index"))


@app.route("/shutdown-app", methods=["POST"])
def shutdown_app():
    """
    Terminates the whole running Spark process — not just a call.
    Refuses to run while a call is in progress (stop the call first),
    so the process is never killed mid-conversation before that call's
    data gets finalized. Uses os._exit() rather than any Werkzeug-
    specific shutdown mechanism — the old environ['werkzeug.server.
    shutdown'] trick has been deprecated for years and isn't reliably
    available in current Werkzeug versions. os._exit() is blunt but
    works regardless of server internals, which matters more here.
    """
    with _call_lock:
        if _call_in_progress:
            flash("Stop the current call before shutting down.")
            return redirect(url_for("index"))

    def _delayed_exit():
        # Small delay so this response actually reaches the browser
        # before the process disappears out from under it.
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_delayed_exit, daemon=True).start()
    return render_template("shutdown.html")


@app.route("/live-transcript")
def live_transcript():
    """
    Polled by the page while a call is running. Reads straight off the
    live ChatSession's transcript list, which the background call-
    thread is appending to turn by turn as the conversation happens —
    no separate live-tracking needed, this is the same object the call
    itself is using.
    """
    with _call_lock:
        in_progress = _call_in_progress
        transcript = (
            list(_voice_session.chat.transcript)
            if in_progress and _voice_session is not None
            else []
        )
    return {"in_progress": in_progress, "transcript": transcript}


@app.route("/review-conversations")
def review_conversations():
    """
    Last 5 calls, most recent first, each with its full transcript.
    Standing in for a proper History page for now — that was meant to
    be organized by contact number, but there's nothing to categorize
    by yet since no telephony integration exists to supply one.
    """
    return render_template("review_conversations.html", calls=_get_recent_calls(limit=5))


@app.route("/page/<name>")
def placeholder(name):
    """
    Temporary stand-in for the Dashboard/History/Settings sub-pages
    (Review Conversations, Dataset, Set Voice, Guidelines, History,
    message api, call api) — built out in a later pass. Keeps every nav
    link in the mockup clickable and honest about being unfinished,
    rather than 404ing.
    """
    titles = {
        "review-conversations": "Review Conversations",
        "dataset": "Dataset",
        "set-voice": "Set Voice",
        "guidelines": "Guidelines",
        "history": "History",
        "message-api": "Message API",
        "call-api": "Call API",
    }
    title = titles.get(name, name.replace("-", " ").title())
    return render_template("placeholder.html", title=title)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)