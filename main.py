"""
Spark — AI sales caller. CLI entry point.

Run with:
    python main.py
"""
from app.conversation.call_runner import run_call

if __name__ == "__main__":
    run_call()