"""
Connects the pieces of the post-call pipeline that already existed but
weren't wired together: ChatSession's live-tracked state (name, org,
callback time, status — reliable, since it comes from the actual
deterministic state machine, including any mid-call corrections) +
parser.py's transcript-based extraction (requirements, quantity, brands,
location — fields nothing else currently tracks) + call_storage.py's
SQLite persistence.

Call finalize_call(chat_session) once a call has genuinely ended.
"""
import json

from app.parser import parse_transcript
from app.call_storage import init_db, save_call
from app.conversation.stages import ConversationStage


def _resolve_status(chat_session) -> str:
    """
    Prefer the actual state machine over parser.py's own extract_status(),
    which guesses status by pattern-matching transcript text (e.g.
    checking for the literal substring "our specialist will contact
    you"). That's fragile — it silently breaks if that confirmation
    sentence's wording ever changes. controller.stage is the real,
    structured source of truth for how the call actually ended.
    """
    stage = chat_session.controller.stage
    memory = chat_session.controller.memory

    if stage == ConversationStage.REJECTION:
        return "rejected"

    if stage == ConversationStage.CLOSING:
        return "callback_scheduled" if memory.callback_time else "closed_no_callback"

    if memory.callback_requested:
        # Customer agreed to a specialist, but the call ended before
        # name/org/time collection finished — worth a human following up.
        return "scheduling_incomplete"

    return "incomplete"


def finalize_call(chat_session, phone=None):
    """
    Runs once, when a call has ended. Writes the consolidated JSON
    summary (chat_session.save_summary()) and a structured row in
    data/spark.db, merging the reliable live-tracked fields with the
    transcript-only extractions parser.py provides.

    `phone` is a placeholder for once the telephony integration exists —
    nothing currently tracks a caller's phone number.
    """
    chat_session.save_summary()

    parsed = parse_transcript(chat_session._transcript_path)

    record = {
        "phone": phone,
        # Prefer chat_session's live-tracked values — they reflect any
        # mid-call correction ("actually it's Alex, not Ali"), which a
        # transcript re-scan has no way to know happened. Fall back to
        # parser.py's guess only if chat_session never captured it.
        "name": chat_session.customer_name or parsed.get("name"),
        "company": chat_session.customer_organization or parsed.get("company"),
        "sector": None,  # not derived anywhere yet — placeholder for later
        "requirements": json.dumps(parsed.get("requirements") or [], ensure_ascii=False),
        "quantity": parsed.get("quantity"),
        "brands": json.dumps(parsed.get("brands") or [], ensure_ascii=False),
        "location": json.dumps(parsed.get("location") or [], ensure_ascii=False),
        "callback_time": chat_session.customer_callback_time or parsed.get("callback_time"),
        "status": _resolve_status(chat_session),
        "transcript": json.dumps(chat_session.transcript, ensure_ascii=False),
    }

    init_db()
    save_call(record)

    print(
        f"[CALL] Finalized — status: {record['status']}, "
        f"name: {record['name']}, company: {record['company']}, "
        f"callback: {record['callback_time']}"
    )

    return record