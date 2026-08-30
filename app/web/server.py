"""
SparkSCT web UI — a local-only Flask app. No auth, no production WSGI
server needed: each device runs its own instance against its own local
resources (Ollama, mic, DB), so this is meant to be reached only via
127.0.0.1 on the same machine.
"""
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd
import yaml

from flask import Flask, render_template, redirect, url_for, flash, request

from app.conversation.call_runner import run_call
from app.conversation.voice_session import VoiceSession
from app.call_storage import DB_PATH, delete_call as _delete_call, add_numbers_to_queue, get_queue, set_queue_status, get_next_pending, mark_stale_calls_as_failed, clear_queue, add_number_and_get_id
from telephony.dialer import (
    place_call,
    ExotelConfigError,
    get_exotel_settings,
    set_exotel_settings,
    EXOTEL_FIELDS,
    normalize_phone,
)
from app.llm.prompts import SYSTEM_PROMPT, OPENING_MESSAGE
from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT

app = Flask(__name__)
app.secret_key = "spark-local-dev"  # local-only app, no real auth surface

SPARK_VERSION = "1.0"


@app.context_processor
def inject_footer_info():
    """Makes spark_version and current_year available in every
    template automatically — base.html's footer uses these — without
    needing every single route's render_template() call to pass them
    individually."""
    import datetime
    return {"spark_version": SPARK_VERSION, "current_year": datetime.date.today().year}

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

# Same three files two_way_test.py (the separate telephony process)
# uses — this is the entire cross-process coordination mechanism for
# batch dialing, no sockets/IPC needed for something this infrequent.
BATCH_ACTIVE_PATH = Path("data/batch_active")
BATCH_STOP_SIGNAL_PATH = Path("data/batch_stop_signal")
ACTIVE_CALL_ID_PATH = Path("data/active_call_id.txt")

# Generous buffer over one call (~5 min at most) plus the 2.5-min pause
# between calls — long enough to never mistake a genuinely running
# batch for a stuck one.
BATCH_STALE_SECONDS = 600


def _fetch_calls(status=None):
    """
    Base query — all calls (or filtered to one status), most recent
    first, transcript parsed from JSON into a list. Every other query
    helper builds on this one, instead of each repeating its own
    near-identical SQL + parsing.
    """
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if status:
        rows = conn.execute(
            "SELECT * FROM calls WHERE status = ? ORDER BY id DESC", (status,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM calls ORDER BY id DESC").fetchall()
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


def _get_recent_calls(limit=5):
    """Most recent calls — shared by the main page's "Last call" box
    (limit=1) and Review Conversations (limit=5)."""
    return _fetch_calls()[:limit]


def _get_latest_call():
    """
    The most recent call — this only exists because finalize_call() ran,
    which only happens when a call genuinely finished. Used as the "did
    a call actually end" confirmation on the page, grounded in real
    saved data rather than an in-memory flag.
    """
    calls = _get_recent_calls(limit=1)
    return calls[0] if calls else None


def _get_calls_by_status():
    """
    All calls, grouped into the three actual outcome buckets
    call_finalizer.py produces: scheduled (callback_scheduled), rejected
    (rejected), and incomplete (everything else — closed_no_callback,
    scheduling_incomplete, incomplete — calls that ended without a
    clean resolution).
    """
    buckets = {"scheduled": [], "rejected": [], "incomplete": []}
    for call in _fetch_calls():
        if call["status"] == "callback_scheduled":
            buckets["scheduled"].append(call)
        elif call["status"] == "rejected":
            buckets["rejected"].append(call)
        else:
            buckets["incomplete"].append(call)
    return buckets


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
    # Run here too, not just on /call-queue — this is the page you'd
    # actually be looking at between calls during a real batch, and
    # the stale-call safety net only ever runs when SOMEONE loads a
    # page that checks it. If the whole process dies (crash, manual
    # kill, anything) mid-call, nothing else would ever catch that
    # stuck entry otherwise.
    caught = mark_stale_calls_as_failed()
    if caught:
        # A stuck entry means the batch that was dialing it is
        # effectively dead too — clear this now rather than making
        # the user wait out the separate, longer BATCH_STALE_SECONDS
        # timeout before they can start a new one.
        BATCH_ACTIVE_PATH.unlink(missing_ok=True)
        flash(f"{caught} call(s) stuck on 'calling' for too long — marked as failed.")

    with _call_lock:
        in_progress = _call_in_progress
    return render_template(
        "index.html",
        call_in_progress=in_progress,
        batch_active=BATCH_ACTIVE_PATH.exists(),
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
    # A batch runs in the separate telephony process, so it's tracked
    # via the shared file, not _call_in_progress — check both, since
    # either (or neither) could be true depending on what's running.
    batch_was_active = BATCH_ACTIVE_PATH.exists()
    if batch_was_active:
        BATCH_STOP_SIGNAL_PATH.parent.mkdir(exist_ok=True)
        BATCH_STOP_SIGNAL_PATH.touch()

    with _call_lock:
        in_progress = _call_in_progress
    if in_progress:
        _stop_event.set()

    if not in_progress and not batch_was_active:
        flash("No call is currently in progress.")
        return redirect(url_for("index"))

    if batch_was_active:
        flash("Stopping — the current call will finish, then no further numbers will be dialed.")
    else:
        flash("Stopping — the current exchange will finish, then the call will end.")
    return redirect(url_for("index"))


@app.route("/start-calling-queue", methods=["POST"])
def start_calling_queue():
    """
    Dials the FIRST pending number in Call Queue. Everything after
    that — pausing, dialing the next number, ending the batch — is
    handled entirely by the telephony server (two_way_test.py), which
    already knows the exact instant each call ends. This route's only
    job is starting the chain.
    """
    if BATCH_ACTIVE_PATH.exists():
        # If the flag is genuinely recent, a batch really is running —
        # refuse to start a second one. But if it's old, the batch that
        # set it never properly continued or cleared it (e.g. its very
        # FIRST call never even connected, so two_way_test.py's finally
        # block — the only place that normally clears this — never ran
        # at all). Don't block forever on a flag nothing's coming back
        # to clean up; same staleness-timeout idea as the queue-entry
        # fix, applied to this file instead of a database row.
        age_seconds = time.time() - BATCH_ACTIVE_PATH.stat().st_mtime
        if age_seconds < BATCH_STALE_SECONDS:
            flash("A batch is already running.")
            return redirect(url_for("index"))
        BATCH_ACTIVE_PATH.unlink()

    next_entry = get_next_pending()
    if not next_entry:
        flash("No pending numbers in Call Queue.")
        return redirect(url_for("index"))

    try:
        place_call(next_entry["phone"], get_exotel_settings())
    except ExotelConfigError as e:
        flash(str(e))
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Call failed to place: {e}")
        return redirect(url_for("index"))

    BATCH_ACTIVE_PATH.parent.mkdir(exist_ok=True)
    BATCH_ACTIVE_PATH.touch()
    set_queue_status(next_entry["id"], "calling")
    flash(f"Started calling queue — dialing {next_entry['phone']}.")
    return redirect(url_for("index"))


@app.route("/call-number", methods=["POST"])
def call_single_number():
    """
    Dials one number directly from the homepage — no need to build an
    Excel sheet just to test a single number. Deliberately NOT gated
    on batch state (unlike start_calling_queue) since this is a one-off
    dial, not part of a sequence — same reasoning as Call Queue's
    per-row Call button. Added to Call Queue and marked 'calling'
    immediately, so it's tracked and shows up in the Calling/Done/
    Failed buckets like any other call, rather than bypassing tracking.
    """
    raw_phone = request.form.get("phone", "").strip()
    if not raw_phone:
        flash("Enter a phone number to call.")
        return redirect(url_for("index"))

    phone = normalize_phone(raw_phone)

    try:
        place_call(phone, get_exotel_settings())
    except ExotelConfigError as e:
        flash(str(e))
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Call failed to place: {e}")
        return redirect(url_for("index"))

    queue_id = add_number_and_get_id(phone)
    set_queue_status(queue_id, "calling")
    flash(f"Calling {phone}.")
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
    Polled by the page while a call is running. Checks for an active
    PHONE call first — that ChatSession lives in the separate
    telephony process, not this one, so it's tracked by tailing the
    same transcripts/{id}.jsonl file chat_session.py already writes to
    incrementally, using ACTIVE_CALL_ID_PATH to know which one. Falls
    back to the original local-mic behavior (reading straight off the
    live ChatSession object) unchanged if no phone call is active.
    """
    if ACTIVE_CALL_ID_PATH.exists():
        call_id = ACTIVE_CALL_ID_PATH.read_text().strip()
        if call_id:
            transcript_path = Path("transcripts") / f"{call_id}.jsonl"
            if transcript_path.exists():
                transcript = []
                for line in transcript_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        transcript.append(json.loads(line))
                return {"in_progress": True, "transcript": transcript}

    with _call_lock:
        in_progress = _call_in_progress
        transcript = (
            list(_voice_session.chat.transcript)
            if in_progress and _voice_session is not None
            else []
        )
    return {"in_progress": in_progress, "transcript": transcript}


@app.route("/history")
def history():
    """
    Working callback queue — ONLY scheduled calls (not a general
    history browser, that's what Review Conversations is for). Each
    entry has a checkbox; checking some and confirming clears them out
    once you've actually made those callbacks.
    """
    return render_template("history.html", calls=_fetch_calls(status="callback_scheduled"))


@app.route("/history/confirm-done", methods=["POST"])
def confirm_callbacks_done():
    ids = request.form.getlist("call_ids")
    for call_id in ids:
        _delete_call(int(call_id))
    flash(f"Cleared {len(ids)} completed callback(s).")
    return redirect(url_for("history"))


@app.route("/review-conversations")
def review_conversations():
    """
    Last 5 calls, most recent first, each with its full transcript.
    General browsing — History above is specifically the scheduled-
    callback working queue, this is everything.
    """
    return render_template("review_conversations.html", calls=_get_recent_calls(limit=5))


@app.route("/call-queue")
def call_queue():
    caught = mark_stale_calls_as_failed()
    if caught:
        flash(f"{caught} call(s) stuck on 'calling' for too long — marked as failed.")

    queue = get_queue()
    pending = [q for q in queue if q["status"] == "pending"]
    calling = [q for q in queue if q["status"] == "calling"]
    failed = [q for q in queue if q["status"] == "failed"]
    done = [q for q in queue if q["status"] == "done"]
    return render_template(
        "call_queue.html", pending=pending, calling=calling, failed=failed, done=done
    )


@app.route("/call-queue/upload", methods=["POST"])
def upload_call_queue():
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("No file selected.")
        return redirect(url_for("call_queue"))

    try:
        df = pd.read_excel(file)
    except Exception as e:
        flash(f"Could not read that file: {e}")
        return redirect(url_for("call_queue"))

    # Accept either "phone" or "number" as the column header, case-insensitive.
    col = None
    for candidate in df.columns:
        if str(candidate).strip().lower() in ("phone", "number"):
            col = candidate
            break

    if col is None:
        flash("Couldn't find a 'phone' or 'number' column in that file.")
        return redirect(url_for("call_queue"))

    numbers = [normalize_phone(n) for n in df[col].dropna().tolist()]
    numbers = [n for n in numbers if n]

    if not numbers:
        flash("No phone numbers found in that column.")
        return redirect(url_for("call_queue"))

    add_numbers_to_queue(numbers)
    flash(f"Added {len(numbers)} number(s) to the queue.")
    return redirect(url_for("call_queue"))


@app.route("/call-queue/call/<int:queue_id>", methods=["POST"])
def trigger_call(queue_id):
    phone = request.form.get("phone")
    if not phone:
        flash("No phone number given.")
        return redirect(url_for("call_queue"))

    try:
        place_call(phone, get_exotel_settings())
    except ExotelConfigError as e:
        flash(str(e))
        return redirect(url_for("call_queue"))
    except Exception as e:
        # Leaves status as "pending" on failure — safe to retry, matches
        # how every other retryable action in this app behaves.
        flash(f"Call failed to place: {e}")
        return redirect(url_for("call_queue"))

    set_queue_status(queue_id, "calling")
    flash(f"Calling {phone}.")
    return redirect(url_for("call_queue"))


@app.route("/call-queue/retry/<int:queue_id>", methods=["POST"])
def retry_call(queue_id):
    """Moves a failed entry back to pending, so it shows up for a
    manual Call click or the next batch run — doesn't auto-redial it
    itself, same "explicit action, not silent auto-retry" pattern as
    everything else retryable in this app."""
    set_queue_status(queue_id, "pending")
    flash("Moved back to pending — click Call, or it'll be picked up by the next batch run.")
    return redirect(url_for("call_queue"))


@app.route("/call-queue/clear-all", methods=["POST"])
def clear_all_queue():
    """Deletes every entry in Call Queue, regardless of status.
    Destructive and irreversible — the template confirms with the
    user via a JS dialog before this route is ever hit."""
    clear_queue()
    flash("Call Queue cleared.")
    return redirect(url_for("call_queue"))


SETTINGS_PATH = Path("config/settings.yml")


def _get_available_voices():
    """Scans models/piper/voices/ for downloaded voice files — each
    voice is a .onnx + .onnx.json pair, named by the voice's own name."""
    voices_dir = Path("models/piper/voices")
    if not voices_dir.exists():
        return []
    return sorted(p.stem for p in voices_dir.glob("*.onnx"))


def _get_current_voice():
    if not SETTINGS_PATH.exists():
        return None
    with open(SETTINGS_PATH) as f:
        data = yaml.safe_load(f) or {}
    return data.get("tts", {}).get("voice")


def _set_voice(new_voice):
    with open(SETTINGS_PATH) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("tts", {})["voice"] = new_voice
    with open(SETTINGS_PATH, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


TARGET_OPTIONS = [
    ("employee_id", "Corporates — employee ID solutions"),
    ("software", "Corporates — software (website, app development)"),
    ("education", "Schools and Universities"),
    ("suppliers", "Suppliers"),
    ("healthcare", "Healthcare"),
    ("manufacturing", "Manufacturing"),
]


@app.route("/target-audience")
def target_audience():
    return render_template("target_audience.html", options=TARGET_OPTIONS)


@app.route("/target-audience/save", methods=["POST"])
def save_target_audience():
    # UI only for now — not wired to an actual campaign switch yet.
    choice = request.form.get("target")
    label = dict(TARGET_OPTIONS).get(choice, choice)
    flash(f"Selected: {label}. Not wired to a real campaign switch yet.")
    return redirect(url_for("target_audience"))


@app.route("/guidelines")
def guidelines():
    return render_template(
        "guidelines.html",
        system_prompt=SYSTEM_PROMPT,
        opening_message=OPENING_MESSAGE,
        employee_id_prompt=EMPLOYEE_ID_PROMPT,
    )


@app.route("/call-api")
def call_api():
    return render_template("call_api.html", settings=get_exotel_settings())


@app.route("/call-api/save", methods=["POST"])
def save_call_api():
    values = {field: request.form.get(field, "").strip() for field in EXOTEL_FIELDS}
    set_exotel_settings(values)
    flash("Exotel settings saved.")
    return redirect(url_for("call_api"))


@app.route("/set-voice")
def set_voice():
    return render_template(
        "set_voice.html",
        voices=_get_available_voices(),
        current=_get_current_voice(),
    )


@app.route("/set-voice/save", methods=["POST"])
def save_voice():
    new_voice = request.form.get("voice")
    if not new_voice:
        flash("No voice selected.")
        return redirect(url_for("set_voice"))

    _set_voice(new_voice)
    flash(
        f"Voice set to {new_voice}. This takes effect on the next reload — "
        f"use \"Stop Gemma\" then Start, or restart the app."
    )
    return redirect(url_for("set_voice"))


@app.route("/dataset")
def dataset():
    """
    Simple tabular view of the parsed call data (name, organization,
    callback time), grouped by outcome — scheduled, rejected, or
    incomplete.
    """
    return render_template("dataset.html", buckets=_get_calls_by_status())


@app.route("/delete-call/<int:call_id>", methods=["POST"])
def delete_call_route(call_id):
    """Single-call delete — used by every per-row Delete button on Dataset."""
    _delete_call(call_id)
    flash("Call deleted.")
    return redirect(url_for("dataset"))


@app.route("/dataset/delete-selected", methods=["POST"])
def delete_selected():
    """
    Bulk delete — shared by all three Dataset buckets, since the action
    itself (delete whatever IDs were checked) doesn't actually depend on
    which bucket they came from. Each bucket's form has its own
    confirm() warning text, which is the only real difference between
    them.
    """
    ids = request.form.getlist("call_ids")
    for call_id in ids:
        _delete_call(int(call_id))
    flash(f"Deleted {len(ids)} call(s).")
    return redirect(url_for("dataset"))


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
        "set-voice": "Set Voice",
        "guidelines": "Guidelines",
        "message-api": "Message API",
        "call-api": "Call API",
    }
    title = titles.get(name, name.replace("-", " ").title())
    return render_template("placeholder.html", title=title)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)