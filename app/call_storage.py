import sqlite3
from pathlib import Path

DB_PATH = Path("data/spark.db")

# Matches chat_session.py's TRANSCRIPT_DIR — this is where the per-call
# .jsonl (and _summary.json) files live, so delete_call() can clean
# those up alongside the database row.
TRANSCRIPT_DIR = Path("transcripts")


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id TEXT,
            phone TEXT,
            name TEXT,
            company TEXT,
            sector TEXT,
            requirements TEXT,
            quantity INTEGER,
            brands TEXT,
            location TEXT,
            callback_time TEXT,
            status TEXT,
            transcript TEXT,
            created_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS call_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            uploaded_at TEXT
        )
    """)

    # Migration safety: call_id didn't exist in earlier versions of this
    # table. If the table already exists without it, add the column
    # rather than requiring the database to be deleted and losing
    # existing call history.
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(calls)")}
    if "call_id" not in existing_columns:
        conn.execute("ALTER TABLE calls ADD COLUMN call_id TEXT")

    # Same migration-safety pattern for call_queue — needed to detect
    # calls that got stuck on "calling" because the connection never
    # actually reached the telephony server at all (e.g. a stale tunnel
    # URL), which produces no event for any code to hook into.
    queue_columns = {row[1] for row in conn.execute("PRAGMA table_info(call_queue)")}
    if "calling_started_at" not in queue_columns:
        conn.execute("ALTER TABLE call_queue ADD COLUMN calling_started_at TEXT")

    conn.commit()
    conn.close()


def save_call(data):
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        INSERT INTO calls (
            call_id,
            phone,
            name,
            company,
            sector,
            requirements,
            quantity,
            brands,
            location,
            callback_time,
            status,
            transcript,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        data.get("call_id"),
        data.get("phone"),
        data.get("name"),
        data.get("company"),
        data.get("sector"),
        data.get("requirements"),
        data.get("quantity"),
        data.get("brands"),
        data.get("location"),
        data.get("callback_time"),
        data.get("status"),
        data.get("transcript"),
    ))

    conn.commit()
    conn.close()


def delete_call(call_db_id):
    """
    Deletes a call's database row AND its transcripts/<call_id>.jsonl
    (plus _summary.json, if it exists) together. The transcript file
    becomes pure leftover the instant a call is finalized — nothing
    reads it again afterward, since Dataset/History/Review Conversations
    only ever read from this database — so removing it here is safe and
    is what actually keeps storage clean, rather than just hiding the
    row from the UI while the file lingers on disk.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT call_id FROM calls WHERE id = ?", (call_db_id,)
    ).fetchone()

    conn.execute("DELETE FROM calls WHERE id = ?", (call_db_id,))
    conn.commit()
    conn.close()

    if row and row["call_id"]:
        transcript_file = TRANSCRIPT_DIR / f"{row['call_id']}.jsonl"
        if transcript_file.exists():
            transcript_file.unlink()

        summary_file = TRANSCRIPT_DIR / f"{row['call_id']}_summary.json"
        if summary_file.exists():
            summary_file.unlink()


def add_numbers_to_queue(phone_numbers):
    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        "INSERT INTO call_queue (phone, status, uploaded_at) VALUES (?, 'pending', datetime('now'))",
        [(p,) for p in phone_numbers],
    )
    conn.commit()
    conn.close()


def set_queue_status(queue_id, status):
    conn = sqlite3.connect(DB_PATH)
    if status == "calling":
        # Records WHEN calling started — this timestamp is what lets a
        # stuck entry (connection never reached the telephony server at
        # all, so no code ever runs to update it) get automatically
        # detected and surfaced later, instead of sitting on "calling"
        # forever with no visibility.
        conn.execute(
            "UPDATE call_queue SET status = ?, calling_started_at = datetime('now') WHERE id = ?",
            (status, queue_id),
        )
    else:
        conn.execute("UPDATE call_queue SET status = ? WHERE id = ?", (status, queue_id))
    conn.commit()
    conn.close()


def mark_done_by_phone(phone):
    """Marks the queue entry with this phone number (currently
    'calling') as 'done', once its call has actually finished. Matches
    by phone rather than queue_id since that's what the telephony
    server — a separate process from the one that placed the call — has
    available when a call ends. Returns the number of rows actually
    updated, so a format mismatch (queue stores +91..., Exotel reports
    something slightly different) shows up as a visible 0 instead of
    silently doing nothing."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "UPDATE call_queue SET status = 'done' WHERE phone = ? AND status = 'calling'",
        (phone,),
    )
    conn.commit()
    rows_updated = cursor.rowcount
    conn.close()
    return rows_updated


def mark_failed_by_phone(phone):
    """Same matching logic as mark_done_by_phone, but for a call that
    genuinely errored out (connection dropped mid-call, an exception
    was raised) rather than completing normally — kept as a visibly
    distinct outcome from 'done' rather than conflating the two."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "UPDATE call_queue SET status = 'failed' WHERE phone = ? AND status = 'calling'",
        (phone,),
    )
    conn.commit()
    rows_updated = cursor.rowcount
    conn.close()
    return rows_updated


STALE_CALL_MINUTES = 5


def mark_stale_calls_as_failed():
    """
    Catches the failure mode NOTHING else can catch: the connection to
    the telephony server never happened at all (e.g. a stale tunnel
    URL) — which means handle() in two_way_test.py never even runs, so
    there's no code path that would otherwise mark this entry done or
    failed. Any entry stuck on 'calling' for longer than any real call
    plausibly takes gets moved to 'failed' here instead, whenever the
    Call Queue page is loaded. Returns how many were caught, for an
    optional flash message.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        f"""
        UPDATE call_queue SET status = 'failed'
        WHERE status = 'calling'
        AND calling_started_at IS NOT NULL
        AND calling_started_at < datetime('now', '-{STALE_CALL_MINUTES} minutes')
        """
    )
    conn.commit()
    rows_updated = cursor.rowcount
    conn.close()
    return rows_updated


def get_next_pending():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM call_queue WHERE status = 'pending' ORDER BY id ASC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_queue():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM call_queue ORDER BY id ASC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


# Runs the migration the moment this module is first imported by
# anything (server.py, call_finalizer.py, a delete route, etc.) —
# rather than only when finalize_call() happens to run first. Without
# this, a database created before call_id existed would only get
# migrated once a NEW call finished, so any earlier action (like
# deleting an old row) would fail with "no such column: call_id".
init_db()