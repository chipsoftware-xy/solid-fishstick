"""
Prints the contents of data/spark.db in a readable form.

Usage:
    python view_calls.py            # all calls, summary view
    python view_calls.py --full     # include full transcript per call
"""
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("data/spark.db")


def main():
    show_full = "--full" in sys.argv

    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH} — no calls saved yet.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM calls ORDER BY id DESC").fetchall()
    conn.close()

    if not rows:
        print("No calls in the database yet.")
        return

    print(f"{len(rows)} call(s):\n")

    for row in rows:
        print(f"--- Call #{row['id']} ({row['created_at']}) ---")
        print(f"  Status       : {row['status']}")
        print(f"  Name         : {row['name']}")
        print(f"  Company      : {row['company']}")
        print(f"  Phone        : {row['phone']}")
        print(f"  Callback     : {row['callback_time']}")

        try:
            requirements = json.loads(row["requirements"] or "[]")
        except (json.JSONDecodeError, TypeError):
            requirements = row["requirements"]
        print(f"  Requirements : {requirements}")

        print(f"  Quantity     : {row['quantity']}")

        try:
            brands = json.loads(row["brands"] or "[]")
        except (json.JSONDecodeError, TypeError):
            brands = row["brands"]
        print(f"  Brands       : {brands}")

        try:
            location = json.loads(row["location"] or "[]")
        except (json.JSONDecodeError, TypeError):
            location = row["location"]
        print(f"  Location     : {location}")

        if show_full:
            try:
                transcript = json.loads(row["transcript"] or "[]")
            except (json.JSONDecodeError, TypeError):
                transcript = []
            print("  Transcript:")
            for entry in transcript:
                print(f"    [{entry.get('role')}] {entry.get('content')}")

        print()


if __name__ == "__main__":
    main()