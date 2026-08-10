from app.conversation.stages import ConversationStage
from app.conversation.memory import ConversationMemory


class ConversationController:

    def __init__(self):
        self.stage = ConversationStage.GREETING
        self.memory = ConversationMemory()

    def set_stage(self, stage):
        self.stage = stage

    def specialist_requested(self):
        self.memory.wants_specialist = True
        self.stage = ConversationStage.SPECIALIST

    def start_scheduling(self):
        self.memory.callback_requested = True
        self.stage = ConversationStage.SCHEDULING

    def set_callback_time(self, time):
        self.memory.callback_time = time
        self.stage = ConversationStage.CLOSING

    def reject(self):
        self.stage = ConversationStage.REJECTION
        self.memory.conversation_ended = True

    def busy(self):
        self.stage = ConversationStage.BUSY

    def close(self):
        self.stage = ConversationStage.CLOSING
        self.memory.conversation_ended = True