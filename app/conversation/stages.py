from enum import Enum


class ConversationStage(Enum):
    GREETING = "greeting"
    INTRODUCTION = "introduction"
    DISCOVERY = "discovery"
    INTEREST = "interest"
    SPECIALIST = "specialist"
    SCHEDULING = "scheduling"
    CLOSING = "closing"

    REJECTION = "rejection"
    BUSY = "busy"
    HUMAN_HANDOFF = "human_handoff"