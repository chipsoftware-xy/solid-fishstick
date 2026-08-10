class ConversationMemory:

    def __init__(self):
        self.name = None
        self.company = None
        self.requirement = None
        self.quantity = None

        self.interested = False
        self.wants_specialist = False

        self.callback_requested = False
        self.callback_time = None

        self.conversation_ended = False