import sqlite3
from pathlib import Path

DB_PATH = Path("data/spark.db")


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

    conn.commit()
    conn.close()


def save_call(data):
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        INSERT INTO calls (
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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
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