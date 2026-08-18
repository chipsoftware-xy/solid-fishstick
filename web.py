"""
Spark — web UI entry point.

Run with:
    python web.py

Starts the local Flask server and opens your browser to it. Only
reachable from this machine (127.0.0.1) — no networking setup needed on
any of the deployed devices.
"""
import threading
import webbrowser

from app.web.server import app


def _open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Timer(1.0, _open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)