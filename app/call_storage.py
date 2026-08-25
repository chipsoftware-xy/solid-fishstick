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
    conn.execute("UPDATE call_queue SET status = ? WHERE id = ?", (status, queue_id))
    conn.commit()
    conn.close()


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